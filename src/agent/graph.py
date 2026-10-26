"""Pre-visit summary workflow: a LangGraph StateGraph with a human approval step.

    resolve_patient -> gather -> draft -> validate -> prepare_review -> human_review -> finalize
                                   ^          |
                                   +-- retry -+   (citation or format problems, at most 2 attempts)
    resolve_patient -> finalize                   (0 or >1 matches: nothing is drafted)

This is a workflow, not an agent: code decides the path. Design choices, each answering a failure:

- gather runs a *fixed* tool plan. A pre-visit summary always needs the same reads, so the model
  is given no tool that reads or changes data: injected text in a note has nothing to steer.
- draft writes the safety-critical sections (allergies, medications, problems) in code, straight
  from the facts. The model writes only a short narrative, so it cannot drop an allergy.
- validate checks every narrative citation against the refs actually fetched for this patient.
  After two failed attempts the narrative is withheld rather than shown.
- prepare_review freezes the exact text the clinician will see and hashes it; the approval is
  bound to that hash, not to a re-rendering.
- human_review calls interrupt(): the run stops, state is checkpointed, and nothing is released
  until a person approves. There is no edge to finalize that skips it (except not_found).
- finalize appends one hash-chained audit record of refs and hashes, never record content.

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
from src.agent.llm import SYSTEM_PROMPT, Narration, NarrativeFormatError, Narrator, user_message

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

MAX_ATTEMPTS = 2
# Overall deadline for one narrative call, on top of the client's own retries.
NARRATIVE_TIMEOUT_S = float(os.environ.get("COPILOT_LLM_TIMEOUT", "90"))

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

# Socio-economic descriptors that Synthea records as Conditions. Only these move out of the problem
# list, matched by SNOMED code. Everything else stays a clinical problem, including other
# "(finding)" concepts such as "At risk for suicide", "Hemoptysis" or "Victim of intimate partner
# abuse": "(finding)" is a SNOMED concept category, not a social label.
SOCIAL_HISTORY_SNOMED = frozenset({
    "160903007",      # Full-time employment
    "160904001",      # Part-time employment
    "741062008",      # Not in labor force
    "73438004",       # Unemployed
    "224299000",      # Received higher education
    "5251000175109",  # Received certificate of high school equivalency
    "224295006",      # Only received primary school education
    "266948004",      # Has a criminal record
    "224355006",      # Served in armed forces
})

# Problem kinds the model can fix on a retry. "sections" is a bug in our renderer; "model_error" is
# an outage. Neither is retried.
RETRYABLE = frozenset({"citation", "empty", "format"})


class Problem(TypedDict):
    kind: str  # citation | empty | format | model_error | sections
    msg: str


class PrevisitState(TypedDict, total=False):
    synthea_id: str
    patient: dict[str, Any] | None
    not_found_reason: str | None
    facts: dict[str, list[dict[str, Any]]]
    truncated: list[str]  # sections whose complete-list read hit the page cap
    allowed_refs: list[str]
    tool_calls: list[dict[str, Any]]
    injection_flags: int
    sections: str
    narrative: list[str]
    narrative_status: str  # disabled | ok | format_error: ... | error: ...
    model: dict[str, Any] | None  # {"id", "params"} of the narrator that wrote the narrative
    usage: dict[str, int]
    prompt_sha256: str | None
    attempts: int
    problems: list[Problem]
    problem_history: list[list[Problem]]  # problems per attempt, for the audit
    summary: str  # the exact text shown to the reviewer
    summary_sha256: str
    decision: dict[str, Any] | None
    status: str  # released | rejected | not_found | failed
    audit_path: str


def build_graph(call_tool: ToolCaller, narrator: Narrator | None, checkpointer: Any = None):
    async def resolve_patient(state: PrevisitState) -> dict:
        args = {"synthea_id": state["synthea_id"]}
        match = await call_tool("find_patient", args)
        calls = [{"tool": "find_patient", "args": args}]
        if not match.get("found"):
            return {"patient": None, "status": "not_found", "not_found_reason": match.get("reason"), "tool_calls": calls}
        return {"patient": match, "tool_calls": calls}

    async def gather(state: PrevisitState) -> dict:
        pid = state["patient"]["logical_id"]
        calls = [(section, tool, {"patient_id": pid, **args}) for section, (tool, args) in PLAN.items()]
        # TaskGroup: if one read fails, the others are cancelled instead of left running. It wraps
        # the failure in an ExceptionGroup; re-raise the first cause so the audit names it.
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(call_tool(tool, args)) for _, tool, args in calls]
        except ExceptionGroup as group:
            raise group.exceptions[0] from group
        facts: dict[str, list[dict[str, Any]]] = {}
        truncated: list[str] = []
        for (section, tool, _), task in zip(calls, tasks):
            res = task.result()
            if "facts" not in res:
                raise RuntimeError(f"{tool} returned no facts for section {section!r}")
            facts[section] = res["facts"]
            if res.get("truncated"):
                truncated.append(section)
        refs = {state["patient"]["ref"]} | {f["ref"] for items in facts.values() for f in items}
        return {
            "facts": facts,
            "truncated": truncated,
            "allowed_refs": sorted(refs),
            "tool_calls": state["tool_calls"] + [{"tool": t, "args": a} for _, t, a in calls],
            "injection_flags": json.dumps(facts).count(REDACTION),
        }

    async def draft(state: PrevisitState) -> dict:
        attempts = state.get("attempts", 0) + 1
        out: dict[str, Any] = {"attempts": attempts,
                               "sections": render_sections(state["patient"], state["facts"], state.get("truncated", []))}
        if narrator is None:
            return out | {"narrative": [], "narrative_status": "disabled", "model": None, "prompt_sha256": None}
        record = record_text(state["facts"])
        feedback = [p["msg"] for p in state.get("problems", []) if p["kind"] in RETRYABLE]
        out["model"] = {"id": narrator.model_id, "params": narrator.params}
        # Hash of exactly what was sent: system prompt, user turn (with feedback), model and settings.
        out["prompt_sha256"] = _sha256(json.dumps(
            {"system": SYSTEM_PROMPT, "user": user_message(record, feedback), "model": out["model"]}, sort_keys=True))
        try:
            result = await asyncio.wait_for(narrator(record, feedback), timeout=NARRATIVE_TIMEOUT_S)
        except NarrativeFormatError as exc:
            return out | {"narrative": [], "narrative_status": f"format_error: {str(exc)[:200]}"}
        except Exception as exc:  # timeout, throttled past retries, auth, network: fail closed, still reviewable
            return out | {"narrative": [], "narrative_status": f"error: {type(exc).__name__}: {str(exc)[:200]}"}
        narration = result if isinstance(result, Narration) else Narration(list(result))
        usage = {k: state.get("usage", {}).get(k, 0) + v for k, v in narration.usage.items()}
        return out | {"narrative": narration.sentences, "narrative_status": "ok", "usage": usage}

    async def validate(state: PrevisitState) -> dict:
        problems: list[Problem] = []
        status = state["narrative_status"]
        if status == "ok":
            problems = [{"kind": "citation", "msg": m} for m in check_citations(state["narrative"], set(state["allowed_refs"]))]
            if not state["narrative"]:
                problems.append({"kind": "empty", "msg": "narrative is empty"})
        elif status.startswith("format_error"):
            problems = [{"kind": "format", "msg": status}]
        elif status.startswith("error"):
            problems = [{"kind": "model_error", "msg": status}]
        # The code-written sections must carry every allergy and active medication. A miss here is
        # a bug in render_sections, not something a retry fixes, so it is reported, not looped on.
        for section in ("allergies", "medications"):
            for f in state["facts"][section]:
                if f"[{f['ref']}]" not in state["sections"]:
                    problems.append({"kind": "sections", "msg": f"{f['ref']} missing from rendered {section}"})
        return {"problems": problems, "problem_history": state.get("problem_history", []) + [problems]}

    def after_validate(state: PrevisitState) -> str:
        retryable = [p for p in state["problems"] if p["kind"] in RETRYABLE]
        return "draft" if retryable and state["attempts"] < MAX_ATTEMPTS else "prepare_review"

    async def prepare_review(state: PrevisitState) -> dict:
        # Frozen here, before the pause, so the approved text cannot drift (e.g. an age computed
        # from today's date) between review and audit.
        summary = compose(state)
        return {"summary": summary, "summary_sha256": _sha256(summary)}

    async def human_review(state: PrevisitState) -> dict:
        # Everything before interrupt() runs again on resume, so it must have no side effects.
        # A resume value is stored against the interrupt() call that received it and replayed on
        # every later resume. Raising on an invalid decision would therefore poison the thread for
        # good; instead an invalid decision triggers a *new* interrupt carrying the error.
        payload = {
            "summary": state["summary"],
            "summary_sha256": state["summary_sha256"],
            "problems": state["problems"],
            "injection_flags": state["injection_flags"],
            "truncated": state.get("truncated", []),
            "narrative_withheld": narrative_withheld(state),
            "error": None,
        }
        while True:
            decision = interrupt(payload)
            if (error := decision_error(decision)) is None:
                return {"decision": decision}
            payload = {**payload, "error": error}

    async def finalize(state: PrevisitState) -> dict:
        decision = state.get("decision")
        status = state.get("status") or ("released" if decision and decision["action"] == "approve" else "rejected")
        path = append_audit(audit_record(state, status))
        return {"status": status, "audit_path": str(path)}

    g = StateGraph(PrevisitState)
    for name, fn in [("resolve_patient", resolve_patient), ("gather", gather), ("draft", draft),
                     ("validate", validate), ("prepare_review", prepare_review),
                     ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve_patient")
    g.add_conditional_edges("resolve_patient", lambda s: "finalize" if s.get("status") == "not_found" else "gather",
                            {"gather": "gather", "finalize": "finalize"})
    g.add_edge("gather", "draft")
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", after_validate, {"draft": "draft", "prepare_review": "prepare_review"})
    g.add_edge("prepare_review", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def decision_error(decision: Any) -> str | None:
    """Why a review decision is unusable, or None if it is valid."""
    if not isinstance(decision, dict) or decision.get("action") not in ("approve", "reject"):
        return "decision must be {'action': 'approve'|'reject', 'reviewer': ..., 'reason': ...}"
    if decision["action"] == "reject" and not str(decision.get("reason") or "").strip():
        return "a rejection needs a reason: it is the feedback that improves the next draft"
    return None


# ----------------------------------------------------------------------------------------- audit
def audit_path() -> Path:
    return Path(os.environ.get("COPILOT_AUDIT", "data/audit/previsit.jsonl"))


def audit_record(state: PrevisitState, status: str, error: str | None = None) -> dict[str, Any]:
    """References, hashes and decisions only. Never record content: the audit must not become a
    second copy of the chart."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthea_id": state.get("synthea_id"),
        "patient_ref": (state.get("patient") or {}).get("ref"),
        "not_found_reason": state.get("not_found_reason"),
        "tool_calls": state.get("tool_calls", []),
        "refs_retrieved": len(state.get("allowed_refs", [])),
        "truncated_sections": state.get("truncated", []),
        "model": state.get("model"),
        "usage": state.get("usage", {}),
        "prompt_sha256": state.get("prompt_sha256"),
        "narrative_sha256": _sha256("\n".join(state["narrative"])) if state.get("narrative") else None,
        "summary_sha256": state.get("summary_sha256"),
        "attempts": state.get("attempts", 0),
        "narrative_status": state.get("narrative_status"),
        "narrative_withheld": narrative_withheld(state) if state.get("problems") is not None else None,
        "problems": state.get("problems", []),
        "problem_history": state.get("problem_history", []),
        "injection_flags": state.get("injection_flags", 0),
        "decision": state.get("decision"),
        "status": status,
        "error": error,
    }


def append_audit(record: dict[str, Any], path: Path | None = None) -> Path:
    """Append one record, chained to the previous one by hash, so an edited or deleted line breaks
    the chain (see verify_audit_chain). Single writer: an API with concurrent runs needs a database
    or write-once object storage instead."""
    path = path or audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    record = {**record, "prev_sha256": _sha256(lines[-1]) if lines else "0" * 64}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def verify_audit_chain(path: Path | None = None) -> list[str]:
    """Return problems; an empty list means no record was edited, removed or reordered."""
    path = path or audit_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    problems, prev = [], "0" * 64
    for i, line in enumerate(lines, 1):
        if json.loads(line).get("prev_sha256") != prev:
            problems.append(f"line {i}: chain broken (previous record edited, removed or reordered)")
        prev = _sha256(line)
    return problems


def audit_failure(state: PrevisitState, exc: BaseException) -> Path:
    """A run that crashes still leaves an audit record: failed attempts are part of the trail."""
    return append_audit(audit_record(state, "failed", error=f"{type(exc).__name__}: {str(exc)[:300]}"))


# ---------------------------------------------------------------------------------------- render
def narrative_withheld(state: PrevisitState) -> bool:
    return state.get("narrative_status") == "ok" and any(p["kind"] in RETRYABLE for p in state.get("problems", []))


def render_sections(patient: dict[str, Any], facts: dict[str, list[dict[str, Any]]], truncated: list[str] = ()) -> str:
    def item(f: dict[str, Any], extra: str = "") -> str:
        when = f" ({f['date']})" if f.get("date") else ""
        return f"- {_one_line(f['text']) or '(no display text)'}{_no_breaks(extra)}{when} [{f['ref']}]"

    def incomplete(section: str) -> list[str]:
        return [f"- **List may be incomplete: more {section} exist than could be retrieved.**"] if section in truncated else []

    age = _age(patient.get("birth_date"), patient.get("deceased_date"))
    lines = [f"## {_one_line(patient.get('name') or '') or 'Unnamed patient'}",
             f"{patient.get('gender') or 'sex unknown'}, {age if age is not None else '?'} y"
             f"{' at death' if patient.get('deceased') else ''}, "
             f"born {patient.get('birth_date') or '?'} [{patient['ref']}]"]
    if patient.get("deceased"):
        lines.append(f"**Record marks this patient as deceased ({patient.get('deceased_date') or 'date not recorded'}).**")

    lines += ["", "### Allergies and intolerances", *incomplete("allergies")]
    if facts["allergies"]:
        for f in facts["allergies"]:
            d = f["detail"]
            rx = ", ".join(r for r in d.get("reactions", []) if r)
            extra = (f", criticality {d['criticality']}" if d.get("criticality") else "") + (f", reaction: {rx}" if rx else "")
            # Anything other than confirmed-and-active is shown, labelled, never silently dropped.
            flags = [s for s in (d.get("verification"), d.get("clinical_status")) if s and s not in ("confirmed", "active")]
            extra += f" — {', '.join(flags).upper()}" if flags else ""
            lines.append(item(f, extra))
    else:
        # Absence of a record is not absence of allergy; never say "no allergies".
        lines.append("- None recorded. This does not mean the patient has no allergies.")

    lines += ["", "### Active medications", *incomplete("medications")]
    lines += [item(f, f" — {f['detail']['dosage']}" if f["detail"].get("dosage") else "") for f in facts["medications"]] \
        or ["- None recorded as active."]

    social = [f for f in facts["conditions"] if f["detail"].get("snomed") in SOCIAL_HISTORY_SNOMED]
    problems = [f for f in facts["conditions"] if f not in social]
    lines += ["", "### Active problems", *incomplete("conditions")]
    lines += [item(f) for f in problems] or ["- None recorded as active."]
    if social:
        lines += ["", "### Social history (employment, education, legal, military)"]
        lines += [item(f) for f in social]

    lines += ["", "### Recent results"]
    lines += [item(f, f": {f['detail'].get('value', '')}") for f in facts["labs"] + facts["vitals"]] or ["- None recorded."]

    lines += ["", "### Recent encounters"]
    lines += [item(f, f", {f['detail']['reason']}" if f["detail"].get("reason") else "") for f in facts["encounters"]] \
        or ["- None recorded."]
    return "\n".join(lines)


def record_text(facts: dict[str, list[dict[str, Any]]]) -> str:
    """The record as the model sees it: one line per fact, citation first. Fenced in llm.user_message."""
    lines = []
    for section, items in facts.items():
        for f in items:
            detail = "; ".join(f"{k}: {v}" for k, v in f.get("detail", {}).items() if v not in (None, "", []) and k != "snomed")
            when = f" | {f['date']}" if f.get("date") else ""
            lines.append(f"[{f['ref']}] {section}: {f['text']}{when}" + (f" | {detail}" if detail else ""))
    return "\n".join(lines)


def compose(state: PrevisitState) -> str:
    parts = [state["sections"], "", "### Narrative"]
    status = state.get("narrative_status", "disabled")
    if status == "disabled":
        parts.append("_No model configured (COPILOT_LLM=none): sections above are written from the record by code._")
    elif status.startswith(("error", "format_error")):
        parts.append(f"_Narrative unavailable ({status.split(':')[0]}). Sections above are complete without it._")
    elif narrative_withheld(state):
        parts.append(f"_Narrative withheld: it failed citation checks after {state['attempts']} attempts._")
    else:
        parts += [_one_line(s) for s in state["narrative"]]
    parts += ["", "_Synthetic data. Not for clinical use. Draft until approved._"]
    return "\n".join(parts)


def _one_line(text: str) -> str:
    """Display encoding: record text renders as one line, so it cannot start a fake heading or
    list item in a Markdown view."""
    return " ".join((text or "").split())


def _no_breaks(text: str) -> str:
    """Like _one_line but keeps the leading space of a fragment such as ' — Take as needed.'"""
    return (text or "").replace("\r", " ").replace("\n", " ")


def _age(birth: str | None, until: str | None = None) -> int | None:
    if not birth:
        return None
    b = date.fromisoformat(birth[:10])
    t = date.fromisoformat(until[:10]) if until else date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
