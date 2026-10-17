"""Pre-visit summary agent: a LangGraph StateGraph with a human approval step.

    resolve_patient -> gather -> draft -> validate -> human_review -> finalize
                                   ^          |
                                   +-- retry -+   (citation problems, at most 2 attempts)
    resolve_patient -> finalize                   (0 or >1 matches: nothing is drafted)

Design choices, each answering a specific failure:

- gather runs a *fixed* tool plan. A pre-visit summary always needs the same reads, so the model
  is never given tools: injected text in a note has nothing to steer.
- draft writes the safety-critical sections (problems, medications, allergies) in code, straight
  from the facts. The model writes only a short narrative, so it cannot drop an allergy.
- validate checks every narrative citation against the refs actually fetched for this patient.
  After two failed attempts the narrative is withheld rather than shown.
- human_review calls interrupt(): the run stops, state is checkpointed, and nothing is released
  until a person approves. There is no edge to finalize that skips it (except not_found).
- finalize appends one audit record of refs and hashes, never record content.

The graph depends on two injected callables, not on MCP or a provider: `call_tool(name, args)`
and an optional `narrator`. The CLI passes an MCP session and Bedrock; tests pass fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agent.guards import REDACTION, check_citations
from src.agent.llm import SYSTEM_PROMPT, Narrator, model_id

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

MAX_ATTEMPTS = 2

# section -> (tool, args). The whole plan, fixed in code.
PLAN: dict[str, tuple[str, dict[str, Any]]] = {
    "conditions": ("get_conditions", {"active_only": True}),
    "medications": ("get_medications", {}),
    "allergies": ("get_allergies", {}),
    "labs": ("get_observations", {"category": "laboratory", "count": 15}),
    "vitals": ("get_observations", {"category": "vital-signs", "count": 10}),
    "encounters": ("get_encounters", {"count": 5}),
    "notes": ("get_notes", {"count": 2}),
}


class PrevisitState(TypedDict, total=False):
    synthea_id: str
    patient: dict[str, Any] | None
    facts: dict[str, list[dict[str, Any]]]
    allowed_refs: list[str]
    tool_calls: list[dict[str, Any]]
    injection_flags: int
    sections: str
    narrative: list[str]
    narrative_status: str  # disabled | ok | error: ...
    prompt_sha256: str | None
    attempts: int
    problems: list[str]
    problem_history: list[list[str]]  # problems per attempt, for the audit
    decision: dict[str, Any] | None
    status: str  # released | rejected | not_found
    audit_path: str


def build_graph(call_tool: ToolCaller, narrator: Narrator | None, checkpointer: Any = None):
    async def resolve_patient(state: PrevisitState) -> dict:
        args = {"synthea_id": state["synthea_id"]}
        match = await call_tool("find_patient", args)
        calls = [{"tool": "find_patient", "args": args}]
        if not match.get("found"):
            return {"patient": None, "status": "not_found", "tool_calls": calls}
        return {"patient": match, "tool_calls": calls}

    async def gather(state: PrevisitState) -> dict:
        pid = state["patient"]["logical_id"]
        calls = [(section, tool, {"patient_id": pid, **args}) for section, (tool, args) in PLAN.items()]
        results = await asyncio.gather(*(call_tool(tool, args) for _, tool, args in calls))
        facts = {section: res["facts"] for (section, _, _), res in zip(calls, results)}
        refs = {state["patient"]["ref"]} | {f["ref"] for items in facts.values() for f in items}
        return {
            "facts": facts,
            "allowed_refs": sorted(refs),
            "tool_calls": state["tool_calls"] + [{"tool": t, "args": a} for _, t, a in calls],
            "injection_flags": json.dumps(facts).count(REDACTION),
        }

    async def draft(state: PrevisitState) -> dict:
        attempts = state.get("attempts", 0) + 1
        out: dict[str, Any] = {"attempts": attempts, "sections": render_sections(state["patient"], state["facts"])}
        if narrator is None:
            return out | {"narrative": [], "narrative_status": "disabled", "prompt_sha256": None}
        record = record_text(state["facts"])
        out["prompt_sha256"] = _sha256(SYSTEM_PROMPT + record)
        try:
            narrative = await narrator(record, state.get("problems", []))
        except Exception as exc:  # throttled past retries, auth, network: fail closed, still reviewable
            return out | {"narrative": [], "narrative_status": f"error: {type(exc).__name__}: {str(exc)[:200]}"}
        return out | {"narrative": narrative, "narrative_status": "ok"}

    async def validate(state: PrevisitState) -> dict:
        problems: list[str] = []
        if state["narrative_status"] == "ok":
            problems = check_citations(state["narrative"], set(state["allowed_refs"]))
            if not state["narrative"]:
                problems.append("narrative is empty")
        elif state["narrative_status"].startswith("error"):
            problems = [state["narrative_status"]]
        # The code-written sections must carry every allergy and active medication. A miss here is
        # a bug in render_sections, not something a retry fixes, so it is reported, not looped on.
        for section in ("allergies", "medications"):
            for f in state["facts"][section]:
                if f"[{f['ref']}]" not in state["sections"]:
                    problems.append(f"sections: {f['ref']} missing from rendered {section}")
        return {"problems": problems, "problem_history": state.get("problem_history", []) + [problems]}

    def after_validate(state: PrevisitState) -> str:
        retryable = [p for p in state["problems"] if not p.startswith(("sections:", "error"))]
        return "draft" if retryable and state["attempts"] < MAX_ATTEMPTS else "human_review"

    async def human_review(state: PrevisitState) -> dict:
        decision = interrupt({
            "summary": compose(state),
            "problems": state["problems"],
            "injection_flags": state["injection_flags"],
            "narrative_withheld": narrative_withheld(state),
        })
        if not isinstance(decision, dict) or decision.get("action") not in ("approve", "reject"):
            raise ValueError("decision must be {'action': 'approve'|'reject', 'reviewer': ..., 'reason': ...}")
        return {"decision": decision}

    async def finalize(state: PrevisitState) -> dict:
        decision = state.get("decision")
        status = state.get("status") or ("released" if decision and decision["action"] == "approve" else "rejected")
        path = Path(os.environ.get("COPILOT_AUDIT", "data/audit/previsit.jsonl"))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "synthea_id": state["synthea_id"],
            "patient_ref": (state.get("patient") or {}).get("ref"),
            "tool_calls": state.get("tool_calls", []),
            "refs_retrieved": len(state.get("allowed_refs", [])),
            "model_id": model_id() if state.get("narrative_status") not in (None, "disabled") else None,
            "prompt_sha256": state.get("prompt_sha256"),
            "summary_sha256": _sha256(compose(state)) if state.get("sections") else None,
            "attempts": state.get("attempts", 0),
            "narrative_status": state.get("narrative_status"),
            "narrative_withheld": narrative_withheld(state) if state.get("sections") else None,
            "problems": state.get("problems", []),
            "problem_history": state.get("problem_history", []),
            "injection_flags": state.get("injection_flags", 0),
            "decision": decision,
            "status": status,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"status": status, "audit_path": str(path)}

    g = StateGraph(PrevisitState)
    for name, fn in [("resolve_patient", resolve_patient), ("gather", gather), ("draft", draft),
                     ("validate", validate), ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve_patient")
    g.add_conditional_edges("resolve_patient", lambda s: "finalize" if s.get("status") == "not_found" else "gather",
                            {"gather": "gather", "finalize": "finalize"})
    g.add_edge("gather", "draft")
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", after_validate, {"draft": "draft", "human_review": "human_review"})
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------------------- render
def narrative_withheld(state: PrevisitState) -> bool:
    return state.get("narrative_status") == "ok" and any(not p.startswith("sections:") for p in state["problems"])


def render_sections(patient: dict[str, Any], facts: dict[str, list[dict[str, Any]]]) -> str:
    def item(f: dict[str, Any], extra: str = "") -> str:
        when = f" ({f['date']})" if f.get("date") else ""
        return f"- {f['text'] or '(no display text)'}{extra}{when} [{f['ref']}]"

    age = _age(patient.get("birth_date"), patient.get("deceased_date"))
    lines = [f"## {patient.get('name') or 'Unnamed patient'}",
             f"{patient.get('gender') or 'sex unknown'}, {age if age is not None else '?'} y"
             f"{' at death' if patient.get('deceased') else ''}, "
             f"born {patient.get('birth_date') or '?'} [{patient['ref']}]"]
    if patient.get("deceased"):
        lines.append(f"**Record marks this patient as deceased ({patient.get('deceased_date') or 'date not recorded'}).**")

    lines += ["", "### Allergies and intolerances"]
    if facts["allergies"]:
        for f in facts["allergies"]:
            crit = f["detail"].get("criticality")
            rx = ", ".join(r for r in f["detail"].get("reactions", []) if r)
            extra = (f", criticality {crit}" if crit else "") + (f", reaction: {rx}" if rx else "")
            lines.append(item(f, extra))
    else:
        # Absence of a record is not absence of allergy; never say "no allergies".
        lines.append("- None recorded. This does not mean the patient has no allergies.")

    lines += ["", "### Active medications"]
    lines += [item(f, f" — {f['detail']['dosage']}" if f["detail"].get("dosage") else "") for f in facts["medications"]] \
        or ["- None recorded as active."]

    # Synthea codes social circumstances as Conditions with the SNOMED "(finding)" tag (employment,
    # education, criminal record). Listing those as clinical problems is wrong and stigmatising.
    problems = [f for f in facts["conditions"] if not f["text"].endswith("(finding)")]
    findings = [f for f in facts["conditions"] if f["text"].endswith("(finding)")]
    lines += ["", "### Active problems"]
    lines += [item(f) for f in problems] or ["- None recorded as active."]
    if findings:
        lines += ["", "### Social and other findings"]
        lines += [item(f) for f in findings]

    lines += ["", "### Recent results"]
    lines += [item(f, f": {f['detail'].get('value', '')}") for f in facts["labs"] + facts["vitals"]] or ["- None recorded."]

    lines += ["", "### Recent encounters"]
    lines += [item(f, f", {f['detail']['reason']}" if f["detail"].get("reason") else "") for f in facts["encounters"]] \
        or ["- None recorded."]
    return "\n".join(lines)


def record_text(facts: dict[str, list[dict[str, Any]]]) -> str:
    """The record as the model sees it: one line per fact, citation first."""
    lines = []
    for section, items in facts.items():
        for f in items:
            detail = "; ".join(f"{k}: {v}" for k, v in f.get("detail", {}).items() if v not in (None, "", []))
            when = f" | {f['date']}" if f.get("date") else ""
            lines.append(f"[{f['ref']}] {section}: {f['text']}{when}" + (f" | {detail}" if detail else ""))
    return "\n".join(lines)


def compose(state: PrevisitState) -> str:
    parts = [state["sections"], "", "### Narrative"]
    status = state.get("narrative_status", "disabled")
    if status == "disabled":
        parts.append("_No model configured (COPILOT_LLM=none): sections above are written from the record by code._")
    elif status.startswith("error"):
        parts.append(f"_Narrative unavailable ({status}). Sections above are complete without it._")
    elif narrative_withheld(state):
        parts.append(f"_Narrative withheld: it failed citation checks after {state['attempts']} attempts._")
    else:
        parts += state["narrative"]
    parts += ["", "_Synthetic data. Not for clinical use. Draft until approved._"]
    return "\n".join(parts)


def _age(birth: str | None, until: str | None = None) -> int | None:
    if not birth:
        return None
    b = date.fromisoformat(birth[:10])
    t = date.fromisoformat(until[:10]) if until else date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
