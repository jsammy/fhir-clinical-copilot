"""Score the Q&A agent's tool routing against the golden set.

    python -m src.eval.routing_eval                 # needs COPILOT_LLM=bedrock, HAPI and the notes index

For each case it checks three things:
- routed:   the agent used the expected kind of tool (fhir lookup vs notes search), or for "none"
            questions answered "Not in record."
- grounded: the answer passed the citation check (status answered or not_in_record)
- bounded:  no refused tools and the patient was never changed (every call used the pinned id)

Routing accuracy here is a measurement, not a promise: the model chooses tools, so the rate is
reported, tracked over time, and is a release gate for any model or prompt change.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.agent.llm import make_agent_model
from src.agent.mcp_session import fhir_tools, load_dotenv
from src.agent.qa_agent import agent_tool_specs, build_agent

GOLDEN = Path(__file__).with_name("routing_golden.json")
RESULTS = Path("data/eval/routing_results.json")
NOTES_TOOLS = {"search_notes", "get_notes"}


def kind_of(tool: str) -> str:
    return "notes" if tool in NOTES_TOOLS else "fhir"


def score(case: dict, result: dict) -> dict:
    used = [t["tool"] for t in result.get("tool_log", []) if t["outcome"] == "ok"]
    if case["expect"] == "none":
        routed = result["status"] == "not_in_record"
    else:
        routed = case["expect"] in {kind_of(t) for t in used} and ("tool" not in case or case["tool"] in used)
    return {
        "id": case["id"], "expect": case["expect"], "tools": used, "status": result["status"],
        "routed": routed,
        "grounded": result["status"] in ("answered", "not_in_record"),
        "bounded": not any(t["outcome"] == "refused" or t.get("patient_id_ignored") for t in result.get("tool_log", [])),
        "tool_calls": result.get("tool_calls", 0), "usage": result.get("usage", {}),
    }


async def main() -> None:
    load_dotenv()
    model = make_agent_model()
    if model is None:
        raise SystemExit("needs a model: COPILOT_LLM=bedrock")
    cases = json.loads(GOLDEN.read_text())["cases"]
    rows = []
    async with fhir_tools("http://localhost:8080/fhir") as (call_tool, tools):
        agent = build_agent(call_tool, model, agent_tool_specs(tools))
        for case in cases:
            result = await agent.ainvoke({"synthea_id": case["patient"], "question": case["question"]}, {"recursion_limit": 40})
            rows.append(row := score(case, result))
            marks = " ".join(f"{k}={'ok' if row[k] else 'FAIL'}" for k in ("routed", "grounded", "bounded"))
            print(f"{case['id']:22} {marks}  tools={','.join(row['tools']) or '-'}  status={row['status']}")
    n = len(rows)
    summary = {k: f"{sum(r[k] for r in rows)}/{n}" for k in ("routed", "grounded", "bounded")}
    tokens = {k: sum(r["usage"].get(k, 0) for r in rows) for k in ("input_tokens", "output_tokens")}
    print(f"\nrouted {summary['routed']}  grounded {summary['grounded']}  bounded {summary['bounded']}  tokens {tokens}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"model": model.model_id, "summary": summary, "tokens": tokens, "cases": rows}, indent=2))
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
