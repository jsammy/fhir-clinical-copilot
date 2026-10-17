"""Run a pre-visit summary from the terminal, with the clinician approval step.

    python -m src.agent.previsit_cli                                   # demo patient, asks you
    python -m src.agent.previsit_cli --patient <synthea-uuid> --decision approve --reason "checked"
    COPILOT_LLM=bedrock python -m src.agent.previsit_cli               # with the Bedrock narrative

What happens: the MCP server is started as a child process over stdio; its advertised tools must
match the allow-list exactly or nothing runs. The graph runs until human_review pauses it, the
draft is shown, your decision resumes it, and one audit line is written. State lives in a SQLite
checkpoint, so a paused run can be resumed later with --thread.
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

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.guards import TOOL_ALLOW_LIST
from src.agent.llm import make_narrator, model_id

# Alive, 9 active conditions, 4 active medications, 8 recorded allergies.
DEMO_PATIENT = "d477c46c-6e56-e32f-0c28-7356caab1e8b"
ROOT = Path(__file__).resolve().parents[2]


def mcp_client(fhir_base_url: str) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "fhir": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "src.agent.mcp_server"],
            "cwd": str(ROOT),
            "env": {**os.environ, "FHIR_BASE_URL": fhir_base_url, "FASTMCP_LOG_LEVEL": "WARNING"},
        }
    })


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=value lines from the gitignored .env, for settings like AWS_BEARER_TOKEN_BEDROCK that must
    never be committed or typed into a shared terminal. Variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ask_decision(args: argparse.Namespace) -> dict[str, Any]:
    action = args.decision
    while action not in ("approve", "reject"):
        action = input("\nApprove this summary for release? [approve/reject]: ").strip().lower()
    reason = args.reason if args.reason is not None else input("Reason (optional): ").strip()
    return {"action": action, "reviewer": args.reviewer, "reason": reason}


async def run(args: argparse.Namespace) -> int:
    narrator = make_narrator()
    print(f"Model: {model_id() or 'none (code-written sections only)'}")

    async with mcp_client(args.fhir_base_url).session("fhir") as session:
        advertised = {t.name for t in (await session.list_tools()).tools}
        if advertised != TOOL_ALLOW_LIST:
            print(f"Refusing to run: server tools {sorted(advertised ^ TOOL_ALLOW_LIST)} differ from the allow-list")
            return 2

        async def call_tool(name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
            if name not in TOOL_ALLOW_LIST:
                raise PermissionError(f"{name} is not an allowed tool")
            result = await session.call_tool(name, tool_args)
            if result.isError:
                raise RuntimeError(f"{name} failed: {' '.join(getattr(c, 'text', '') for c in result.content)}")
            return result.structuredContent or {}

        Path(args.state_db).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(args.state_db) as saver:
            graph = build_graph(call_tool, narrator, checkpointer=saver)
            thread = args.thread or str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread}}

            if args.thread:
                snapshot = await graph.aget_state(config)
                if not snapshot.next:
                    print(f"Thread {thread} has nothing pending.")
                    return 1
                result = {"__interrupt__": snapshot.tasks[0].interrupts}
            else:
                print(f"Thread: {thread}\n")
                result = await graph.ainvoke({"synthea_id": args.patient}, config)

            if "__interrupt__" in result:
                review = result["__interrupt__"][0].value
                print(review["summary"])
                print(f"\nChecks: {len(review['problems'])} problem(s), "
                      f"{review['injection_flags']} injection-like span(s) redacted from the record")
                for p in review["problems"]:
                    print(f"  - {p}")
                result = await graph.ainvoke(Command(resume=ask_decision(args)), config)

    print(f"\nStatus: {result['status']}   audit: {result.get('audit_path')}")
    return 0 if result["status"] == "released" else 1


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default=DEMO_PATIENT, help="Synthea patient UUID")
    ap.add_argument("--fhir-base-url", default=os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir"))
    ap.add_argument("--decision", choices=["approve", "reject"], help="skip the prompt (scripted runs)")
    ap.add_argument("--reason", help="reason recorded with the decision")
    ap.add_argument("--reviewer", default=getpass.getuser())
    ap.add_argument("--thread", help="resume a paused run by its thread id")
    ap.add_argument("--state-db", default="data/agent_state.db")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
