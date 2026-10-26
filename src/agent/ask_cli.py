"""Ask a free-form question about one patient; the agent chooses which read-only tools to call.

    python -m src.agent.ask_cli "Has his blood pressure improved since he started lisinopril?"
    python -m src.agent.ask_cli --patient <synthea-uuid> "Has she mentioned breathing problems?"

Needs a model (COPILOT_LLM=bedrock) and, for search_notes, the notes index:
    python -m src.retrieval.notes_index build
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.agent.llm import make_agent_model
from src.agent.mcp_session import fhir_tools, load_dotenv
from src.agent.qa_agent import agent_tool_specs, build_agent

DEMO_PATIENT = "d477c46c-6e56-e32f-0c28-7356caab1e8b"  # Leif Abbott


def show(result: dict) -> None:
    print("\nTools the agent chose:")
    for i, t in enumerate(result.get("tool_log", []), 1):
        args = ", ".join(f"{k}={v}" for k, v in t["args"].items()) or "-"
        extra = f" -> {t['results']} result(s)" if "results" in t else ""
        flag = " (patient_id it supplied was ignored)" if t.get("patient_id_ignored") else ""
        print(f"  {i}. {t['tool']}({args}) [{t['outcome']}]{extra}{flag}")
    if not result.get("tool_log"):
        print("  (none)")
    print(f"\nAnswer ({result['status']}):")
    if result.get("answer"):
        for s in result["answer"]:
            print(f"  {s}")
    elif result["status"] == "withheld":
        print("  Withheld: the answer failed citation checks twice.")
        for p in result.get("problems", []):
            print(f"    - {p}")
    elif result["status"] == "step_limit":
        print("  Stopped: the agent kept calling tools after its budget was used up.")
    elif result["status"] == "not_found":
        print("  Patient not found.")
    print(f"\nTool calls: {result.get('tool_calls', 0)}   tokens: {result.get('usage', {})}   audit: {result.get('audit_path')}")
    print("Synthetic data. Not for clinical use.")


async def run(args: argparse.Namespace) -> int:
    model = make_agent_model()
    if model is None:
        print("The Q&A agent needs a model: set COPILOT_LLM=bedrock (see .env.example).")
        return 2
    print(f"Model: {model.model_id}\nQuestion: {args.question}")
    try:
        async with fhir_tools(args.fhir_base_url) as (call_tool, tools):
            agent = build_agent(call_tool, model, agent_tool_specs(tools))
            result = await agent.ainvoke({"synthea_id": args.patient, "question": args.question},
                                         {"recursion_limit": 40})
    except PermissionError as exc:
        print(f"Refusing to run: {exc}")
        return 2
    show(result)
    return 0 if result["status"] in ("answered", "not_in_record") else 1


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--patient", default=DEMO_PATIENT, help="Synthea patient UUID")
    ap.add_argument("--fhir-base-url", default="http://localhost:8080/fhir")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
