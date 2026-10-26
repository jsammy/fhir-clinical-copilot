"""Run a pre-visit summary from the terminal, with the clinician approval step.

    python -m src.agent.previsit_cli                                   # demo patient, asks you
    python -m src.agent.previsit_cli --patient <synthea-uuid> --decision approve --reason "checked"
    python -m src.agent.previsit_cli --thread <id>                     # resume a paused review
    COPILOT_LLM=bedrock python -m src.agent.previsit_cli               # with the Bedrock narrative

What happens: the MCP server is started as a child process over stdio, with a minimal environment
(no cloud credentials); its advertised tools must match the allow-list exactly or nothing runs.
The graph runs until human_review pauses it, the draft is shown, your decision resumes it, and one
audit line is written. A run that crashes is audited as failed.

The SQLite checkpoint holds the full record while a review is pending; it is what lets a paused run
resume later. Once a run is finalized its checkpoints are deleted, so the store does not become a
second copy of the chart. With real data the store would also need encryption at rest.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.agent.graph import audit_failure, build_graph
from src.agent.llm import make_narrator
from src.agent.mcp_session import fhir_tools, load_dotenv  # noqa: F401  (load_dotenv re-exported for tests)

# Alive, 9 active conditions, 4 active medications, 8 recorded allergies.
DEMO_PATIENT = "d477c46c-6e56-e32f-0c28-7356caab1e8b"


def ask_decision(args: argparse.Namespace) -> dict[str, Any]:
    action = args.decision
    while action not in ("approve", "reject"):
        action = input("\nApprove this summary for release? [approve/reject]: ").strip().lower()
    reason = args.reason if args.reason is not None else input("Reason: ").strip()
    while action == "reject" and not reason:
        reason = input("A rejection needs a reason: ").strip()
    return {"action": action, "reviewer": args.reviewer, "reason": reason}


async def pending_review(graph: Any, config: dict) -> dict | None:
    """The interrupt payload of a paused thread, or None if nothing is waiting for review."""
    snapshot = await graph.aget_state(config)
    if not snapshot.next or not snapshot.tasks or not snapshot.tasks[0].interrupts:
        return None
    return snapshot.tasks[0].interrupts[0].value


def show_review(review: dict) -> None:
    if review.get("error"):
        return  # the summary was already shown; only the decision is asked again
    print(review["summary"])
    print(f"\nChecks: {len(review['problems'])} problem(s), "
          f"{review['injection_flags']} injection-like span(s) redacted from the record"
          + (f", incomplete lists: {', '.join(review['truncated'])}" if review.get("truncated") else ""))
    for p in review["problems"]:
        print(f"  - [{p['kind']}] {p['msg']}")


async def run_with_graph(graph: Any, saver: Any, args: argparse.Namespace) -> dict:
    thread = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}}
    try:
        if args.thread:
            review = await pending_review(graph, config)
            if review is None:
                print(f"Thread {thread} has nothing pending.")
                return {"status": "nothing_pending"}
        else:
            print(f"Thread: {thread}\n")
            result = await graph.ainvoke({"synthea_id": args.patient}, config)
            review = result["__interrupt__"][0].value if "__interrupt__" in result else None
        while review is not None:
            show_review(review)
            # ask_decision always returns a dict: Command(resume=None) crashes inside LangGraph 1.0.8.
            result = await graph.ainvoke(Command(resume=ask_decision(args)), config)
            review = result["__interrupt__"][0].value if "__interrupt__" in result else None
            if review is not None:  # the graph refused the decision and asked again
                print(f"\nDecision not accepted: {review['error']}")
                args.decision = args.reason = None
    except Exception as exc:
        state = (await graph.aget_state(config)).values
        print(f"\nRun failed ({type(exc).__name__}: {exc}). Audited as failed: {audit_failure(state, exc)}")
        raise
    # Finalized: the audit has the refs and hashes; the checkpoint's copy of the record goes.
    await saver.adelete_thread(thread)
    return result


async def run(args: argparse.Namespace) -> int:
    narrator = make_narrator()
    print(f"Model: {narrator.model_id if narrator else 'none (code-written sections only)'}")

    try:
        async with fhir_tools(args.fhir_base_url) as (call_tool, _):
            Path(args.state_db).parent.mkdir(parents=True, exist_ok=True)
            async with AsyncSqliteSaver.from_conn_string(args.state_db) as saver:
                result = await run_with_graph(build_graph(call_tool, narrator, checkpointer=saver), saver, args)
    except PermissionError as exc:
        print(f"Refusing to run: {exc}")
        return 2

    if result["status"] == "nothing_pending":
        return 1
    print(f"\nStatus: {result['status']}   audit: {result.get('audit_path')}")
    return 0 if result["status"] == "released" else 1


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default=DEMO_PATIENT, help="Synthea patient UUID")
    ap.add_argument("--fhir-base-url", default=os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir"))
    ap.add_argument("--decision", choices=["approve", "reject"], help="skip the prompt (scripted runs)")
    ap.add_argument("--reason", help="reason recorded with the decision (required to reject)")
    ap.add_argument("--reviewer", default=getpass.getuser(),
                    help="who decided; a real deployment takes this from the authenticated session")
    ap.add_argument("--thread", help="resume a paused run by its thread id")
    ap.add_argument("--state-db", default="data/agent_state.db")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
