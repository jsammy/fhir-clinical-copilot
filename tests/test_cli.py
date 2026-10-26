"""The CLI's run logic with the real graph and fake tools: resume, checkpoint cleanup, failure audit."""

import argparse
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import build_graph
from src.agent.previsit_cli import load_dotenv, pending_review, run_with_graph
from tests.test_graph import base_facts, make_tools


@pytest.fixture(autouse=True)
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("COPILOT_AUDIT", str(path))
    return path


def args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"thread": None, "patient": "abc-123", "decision": "approve",
                                 "reason": "checked", "reviewer": "dr-test", **kw})


async def test_finalized_run_deletes_its_checkpoints() -> None:
    """Issue #17: the checkpoint's copy of the record does not outlive the run."""
    saver = InMemorySaver()
    graph = build_graph(make_tools(base_facts()), None, checkpointer=saver)
    result = await run_with_graph(graph, saver, args(thread=None))
    assert result["status"] == "released"
    assert not list(saver.list(None)), "no checkpoints left after finalize"


async def test_resume_a_paused_thread() -> None:
    """Issue #5: the --thread path."""
    saver = InMemorySaver()
    graph = build_graph(make_tools(base_facts()), None, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "paused-1"}}
    await graph.ainvoke({"synthea_id": "abc-123"}, cfg)
    assert (await pending_review(graph, cfg))["summary"]
    result = await run_with_graph(graph, saver, args(thread="paused-1"))
    assert result["status"] == "released"


async def test_nothing_pending() -> None:
    saver = InMemorySaver()
    graph = build_graph(make_tools(base_facts()), None, checkpointer=saver)
    assert (await run_with_graph(graph, saver, args(thread="never-started")))["status"] == "nothing_pending"


async def test_crashed_run_is_audited_as_failed(audit_file) -> None:
    """Issue #18: failed attempts are part of the trail."""
    saver = InMemorySaver()
    graph = build_graph(make_tools(base_facts(), fail="get_medications"), None, checkpointer=saver)
    with pytest.raises(Exception):
        await run_with_graph(graph, saver, args())
    record = json.loads(audit_file.read_text())
    assert record["status"] == "failed" and "get_medications exploded" in record["error"]


def test_dotenv_keeps_values_with_equals_and_respects_existing(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nKEY_A=abc==\nKEY_B="quoted"\nKEY_C=from-file\nnot a setting\n')
    monkeypatch.delenv("KEY_A", raising=False)
    monkeypatch.delenv("KEY_B", raising=False)
    monkeypatch.setenv("KEY_C", "from-shell")
    load_dotenv(env)
    import os
    assert os.environ["KEY_A"] == "abc==" and os.environ["KEY_B"] == "quoted" and os.environ["KEY_C"] == "from-shell"
