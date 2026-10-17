"""The agent graph with fake tools and a fake narrator: no Docker, no AWS, deterministic."""

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.guards import REDACTION

PATIENT = {"found": True, "ref": "Patient/7", "logical_id": "7", "name": "Test Patient",
           "gender": "female", "birth_date": "1960-01-01", "deceased": False, "deceased_date": None}


def fact(ref: str, text: str, **detail: Any) -> dict[str, Any]:
    return {"ref": ref, "kind": ref.split("/")[0], "text": text, "date": "2024-01-01", "detail": detail}


FACTS = {
    "get_conditions": [fact("Condition/1", "Diabetes mellitus type 2 (disorder)"),
                       fact("Condition/2", "Full-time employment (finding)")],
    "get_medications": [fact("MedicationRequest/3", "Metformin 500 MG Oral Tablet", dosage="")],
    "get_allergies": [fact("AllergyIntolerance/4", "Penicillin", criticality="high", reactions=["Hives"])],
    "get_observations": [fact("Observation/5", "Hemoglobin A1c", value="7.4 %")],
    "get_encounters": [fact("Encounter/6", "General examination", reason="")],
    "get_notes": [fact("DocumentReference/8", f"Routine visit. {REDACTION}")],
}


def make_tools(found: bool = True, log: list | None = None):
    async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if log is not None:
            log.append(name)
        if name == "find_patient":
            return PATIENT if found else {"found": False}
        return {"count": len(FACTS[name]), "facts": FACTS[name]}
    return call_tool


def scripted_narrator(*replies: list[str], seen: list | None = None):
    queue = list(replies)

    async def narrate(record: str, feedback: list[str]) -> list[str]:
        if seen is not None:
            seen.append((record, feedback))
        return queue.pop(0)
    return narrate


@pytest.fixture(autouse=True)
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("COPILOT_AUDIT", str(path))
    return path


CFG = {"configurable": {"thread_id": "t1"}}
APPROVE = {"action": "approve", "reviewer": "dr-test", "reason": "checked"}


async def run_to_review(narrator=None, found=True, log=None):
    graph = build_graph(make_tools(found, log), narrator, checkpointer=InMemorySaver())
    result = await graph.ainvoke({"synthea_id": "abc-123"}, CFG)
    return graph, result


async def test_pauses_for_review_and_releases_on_approve(audit_file) -> None:
    graph, result = await run_to_review()
    assert "__interrupt__" in result, "the graph must stop before release"
    assert not audit_file.exists(), "nothing is audited as released before a decision"

    done = await graph.ainvoke(Command(resume=APPROVE), CFG)
    assert done["status"] == "released"
    record = json.loads(audit_file.read_text().splitlines()[-1])
    assert record["decision"]["reviewer"] == "dr-test"
    assert record["status"] == "released"


async def test_reject_is_recorded_not_released(audit_file) -> None:
    graph, _ = await run_to_review()
    done = await graph.ainvoke(Command(resume={"action": "reject", "reviewer": "dr-test", "reason": "wrong patient"}), CFG)
    assert done["status"] == "rejected"


async def test_malformed_decision_is_refused() -> None:
    graph, _ = await run_to_review()
    with pytest.raises(ValueError):
        await graph.ainvoke(Command(resume={"action": "yes please"}), CFG)


async def test_fixed_tool_plan() -> None:
    log: list[str] = []
    await run_to_review(log=log)
    assert log[0] == "find_patient"
    assert sorted(set(log[1:])) == sorted(["get_conditions", "get_medications", "get_allergies",
                                          "get_observations", "get_encounters", "get_notes"])


async def test_safety_sections_written_by_code() -> None:
    _, result = await run_to_review()
    summary = result["__interrupt__"][0].value["summary"]
    assert "Penicillin, criticality high, reaction: Hives" in summary
    assert "[AllergyIntolerance/4]" in summary and "[MedicationRequest/3]" in summary
    # social findings are not clinical problems
    problems_block = summary.split("### Active problems")[1].split("###")[0]
    assert "employment" not in problems_block
    assert "### Social and other findings" in summary


async def test_empty_allergies_never_claims_none(monkeypatch) -> None:
    monkeypatch.setitem(FACTS, "get_allergies", [])
    _, result = await run_to_review()
    summary = result["__interrupt__"][0].value["summary"]
    assert "does not mean the patient has no allergies" in summary


async def test_injection_flags_counted() -> None:
    _, result = await run_to_review()
    assert result["__interrupt__"][0].value["injection_flags"] == 1


async def test_grounded_narrative_is_shown() -> None:
    narrator = scripted_narrator(["Diabetes remains active with HbA1c 7.4% [Condition/1] [Observation/5]."])
    _, result = await run_to_review(narrator)
    review = result["__interrupt__"][0].value
    assert review["problems"] == [] and not review["narrative_withheld"]
    assert "HbA1c 7.4%" in review["summary"]


async def test_fabrication_retried_with_feedback_then_withheld() -> None:
    seen: list = []
    bad = ["Patient has chronic kidney disease [Condition/999]."]
    narrator = scripted_narrator(bad, bad, seen=seen)
    _, result = await run_to_review(narrator)
    review = result["__interrupt__"][0].value
    assert len(seen) == 2, "one retry, then stop"
    assert seen[0][1] == [] and "Condition/999" in seen[1][1][0], "second attempt gets the problems"
    assert review["narrative_withheld"]
    assert "chronic kidney disease" not in review["summary"]


async def test_retry_can_recover() -> None:
    narrator = scripted_narrator(["Uncited claim."], ["Diabetes is active [Condition/1]."])
    _, result = await run_to_review(narrator)
    review = result["__interrupt__"][0].value
    assert review["problems"] == [] and "Diabetes is active" in review["summary"]


async def test_model_failure_fails_closed() -> None:
    async def broken(record: str, feedback: list[str]) -> list[str]:
        raise TimeoutError("bedrock timed out")
    _, result = await run_to_review(broken)
    review = result["__interrupt__"][0].value
    assert "Narrative unavailable" in review["summary"]
    assert "[AllergyIntolerance/4]" in review["summary"], "safety sections survive a model outage"


async def test_unknown_patient_drafts_nothing(audit_file) -> None:
    _, result = await run_to_review(found=False)
    assert "__interrupt__" not in result
    assert result["status"] == "not_found"
    assert json.loads(audit_file.read_text())["status"] == "not_found"
