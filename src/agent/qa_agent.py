"""Q&A agent: the model decides which read-only tools to call to answer a clinician's question.

    resolve_patient -> agent <-> tools          (model chooses tools; at most MAX_TOOL_CALLS)
                         |
                         v
                       check -> agent           (answer fails citation checks: one retry)
                         |
                         v
                       finish                   (audit; answered | not_in_record | withheld | ...)

Where the pre-visit summary is a workflow (code decides the path, because it always needs the same
reads), this is an agent: "has his blood pressure improved since he started lisinopril?" needs
medications, then vital signs, and the path depends on the question. The model routes between
exact FHIR lookups for coded facts and `search_notes` for free text.

What the model can and cannot do, enforced in code:

- It never chooses the patient. The patient is resolved before the model runs, tool schemas have no
  patient_id, and the executor injects it: a note saying "look up patient 42" has nothing to act on.
- It can call only AGENT_TOOLS (read-only), checked on every call, at most MAX_TOOL_CALLS times.
- Every sentence of its answer must cite a resource that a tool returned in this run; one retry
  with feedback, then the answer is withheld. "Not in record." is a valid answer.
- Tool results go back to the model as fenced data, never as instructions.
- One hash-chained audit record per question: tools called, refs cited, hashes, never content.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.graph import ToolCaller, append_audit
from src.agent.guards import AGENT_TOOLS, REFUSAL_RE, check_citations, fence, split_sentences
from src.agent.llm import AGENT_SYSTEM_PROMPT, AgentModel

MAX_TOOL_CALLS = 6
MAX_ANSWER_ATTEMPTS = 2
BUDGET_NOTICE = ("You have used the tool budget for this question. Answer now from the results above, "
                 "or reply exactly: Not in record.")


class QAState(TypedDict, total=False):
    synthea_id: str
    question: str
    patient: dict[str, Any] | None
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: nodes append, never replace
    refs: list[str]          # citable refs returned by tools in this run
    tool_log: list[dict[str, Any]]
    tool_calls: int
    budget_notice: bool
    attempts: int
    answer: list[str]
    problems: list[str]
    status: str              # answered | not_in_record | withheld | step_limit | not_found
    usage: dict[str, int]
    audit_path: str


def agent_tool_specs(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Model-facing tool definitions, derived from the MCP server's own schemas so there is one
    source of truth, minus find_patient and minus every patient_id parameter."""
    specs = []
    for t in mcp_tools:
        if t.name not in AGENT_TOOLS:
            continue
        schema = copy.deepcopy(t.inputSchema)
        schema.get("properties", {}).pop("patient_id", None)
        schema["required"] = [r for r in schema.get("required", []) if r != "patient_id"]
        specs.append({"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": schema}})
    return specs


def render_result(result: dict[str, Any]) -> str:
    """A tool result as the model sees it: citation first, one fact per line, fenced as data."""
    facts = result.get("facts", [])
    if not facts:
        return "<tool_result>No matching records.</tool_result>"
    lines = []
    for f in facts:
        detail = "; ".join(f"{k}: {v}" for k, v in f.get("detail", {}).items()
                           if v not in (None, "", []) and k not in ("snomed", "score", "matched_by"))
        lines.append(f"[{f['ref']}] {f['text']}" + (f" | {f['date']}" if f.get("date") else "") + (f" | {detail}" if detail else ""))
    if result.get("truncated"):
        lines.append("(list incomplete: more records exist than were returned)")
    return "<tool_result>\n" + fence("\n".join(lines)) + "\n</tool_result>"


def answer_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, list):  # Bedrock returns content blocks: text and tool_use
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return " ".join(str(content).split())


def build_agent(call_tool: ToolCaller, model: AgentModel, tool_specs: list[dict[str, Any]], checkpointer: Any = None):
    async def resolve_patient(state: QAState) -> dict:
        match = await call_tool("find_patient", {"synthea_id": state["synthea_id"]})
        if not match.get("found"):
            return {"patient": None, "status": "not_found"}
        return {"patient": match, "refs": [match["ref"]], "tool_log": [], "tool_calls": 0, "attempts": 0,
                "messages": [HumanMessage(state["question"])]}

    async def agent(state: QAState) -> dict:
        reply = await model([SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]], tool_specs)
        usage = dict(state.get("usage", {}))
        for k, v in (getattr(reply, "usage_metadata", None) or {}).items():
            if k in ("input_tokens", "output_tokens"):
                usage[k] = usage.get(k, 0) + v
        return {"messages": [reply], "usage": usage}

    def after_agent(state: QAState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "check"
        if state["tool_calls"] < MAX_TOOL_CALLS:
            return "tools"
        return "finish" if state.get("budget_notice") else "budget"

    async def tools(state: QAState) -> dict:
        pid = state["patient"]["logical_id"]
        refs, log, used, out = set(state["refs"]), list(state["tool_log"]), state["tool_calls"], []
        for call in state["messages"][-1].tool_calls:
            entry: dict[str, Any] = {"tool": call["name"], "args": _loggable(call.get("args", {}))}
            if call["name"] not in AGENT_TOOLS:
                content, entry["outcome"] = f"Refused: {call['name']} is not an available tool.", "refused"
            elif used >= MAX_TOOL_CALLS:
                content, entry["outcome"] = "Skipped: tool budget used up.", "skipped"
            else:
                args = dict(call.get("args") or {})
                if "patient_id" in args:  # the schema has no such field; a model inventing one is ignored
                    entry["patient_id_ignored"] = True
                args["patient_id"] = pid  # pinned by code, every call
                used += 1
                try:
                    result = await call_tool(call["name"], args)
                    refs |= {f["ref"] for f in result.get("facts", [])}
                    content, entry["outcome"], entry["results"] = render_result(result), "ok", len(result.get("facts", []))
                except Exception as exc:  # bad arguments, server error: tell the model, let it adjust
                    content, entry["outcome"] = f"Error: {str(exc)[:300]}", "error"
            log.append(entry)
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out, "refs": sorted(refs), "tool_log": log, "tool_calls": used}

    async def budget(state: QAState) -> dict:
        # Answer the pending tool calls (the API requires it), then tell the model to stop.
        skipped = [ToolMessage(content="Skipped: tool budget used up.", tool_call_id=c["id"], name=c["name"])
                   for c in state["messages"][-1].tool_calls]
        return {"messages": [*skipped, HumanMessage(BUDGET_NOTICE)], "budget_notice": True}

    async def check(state: QAState) -> dict:
        text = answer_text(state["messages"][-1])
        sentences = split_sentences([text]) if text else []
        problems = check_citations(sentences, set(state["refs"])) if sentences else ["the answer is empty"]
        attempts = state["attempts"] + 1
        if not problems:
            status = "not_in_record" if len(sentences) == 1 and REFUSAL_RE.match(sentences[0]) else "answered"
            return {"answer": sentences, "problems": [], "attempts": attempts, "status": status}
        if attempts < MAX_ANSWER_ATTEMPTS:
            feedback = "Your answer was rejected. Fix these problems and answer again:\n- " + "\n- ".join(problems)
            return {"messages": [HumanMessage(feedback)], "problems": problems, "attempts": attempts}
        return {"answer": [], "problems": problems, "attempts": attempts, "status": "withheld"}

    def after_check(state: QAState) -> str:
        return "finish" if state.get("status") else "agent"

    async def finish(state: QAState) -> dict:
        status = state.get("status") or "step_limit"
        cited = sorted({r for s in state.get("answer", []) for r in state.get("refs", []) if f"[{r}]" in s})
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": "qa",
            "synthea_id": state["synthea_id"],
            "patient_ref": (state.get("patient") or {}).get("ref"),
            "question_sha256": _sha256(state["question"]),
            "model": {"id": model.model_id, "params": model.params},
            "tool_log": state.get("tool_log", []),
            "tool_calls": state.get("tool_calls", 0),
            "refs_cited": cited,
            "answer_sha256": _sha256("\n".join(state["answer"])) if state.get("answer") else None,
            "attempts": state.get("attempts", 0),
            "problems": state.get("problems", []),
            "usage": state.get("usage", {}),
            "status": status,
        }
        path = append_audit(record, Path(os.environ.get("COPILOT_QA_AUDIT", "data/audit/qa.jsonl")))
        return {"status": status, "audit_path": str(path)}

    g = StateGraph(QAState)
    for name, fn in [("resolve_patient", resolve_patient), ("agent", agent), ("tools", tools),
                     ("budget", budget), ("check", check), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve_patient")
    g.add_conditional_edges("resolve_patient", lambda s: "finish" if s.get("status") == "not_found" else "agent",
                            {"agent": "agent", "finish": "finish"})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "budget": "budget", "check": "check", "finish": "finish"})
    g.add_edge("tools", "agent")
    g.add_edge("budget", "agent")
    g.add_conditional_edges("check", after_check, {"agent": "agent", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def _loggable(args: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments for the audit. Free-text queries can carry clinical content: hashed, not stored."""
    return {k: (f"sha256:{_sha256(str(v))[:16]}" if k == "query" else v) for k, v in (args or {}).items()}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
