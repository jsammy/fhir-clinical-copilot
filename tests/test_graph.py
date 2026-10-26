"""The workflow graph with fake tools and a fake narrator: no Docker, no AWS, deterministic."""

import asyncio
import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agent import graph as graph_mod
from src.agent.graph import build_graph, verify_audit_chain
from src.agent.guards import REDACTION
from src.agent.llm import Narration, NarrativeFormatError

PATIENT = {"found": True, "ref": "Patient/7", "logical_id": "7", "name": "Test Patient",
           "gender": "female", "birth_date": "1960-01-01", "deceased": False, "deceased_date": None}


def fact(ref: str, text: str, **detail: Any) -> dict[str, Any]:
    return {"ref": ref, "kind": ref.split("/")[0], "text": text, "date": "2024-01-01", "detail": detail}


def base_facts() -> dict[str, list[dict[str, Any]]]:
    return {
        "get_conditions": [fact("Condition/1", "Diabetes mellitus type 2 (disorder)", snomed="44054006"),
                           fact("Condition/2", "Full-time employment (finding)", snomed="160903007"),
                           fact("Condition/9", "At risk for suicide (finding)", snomed="225444004")],
        "get_medications": [fact("MedicationRequest/3", "Metformin 500 MG Oral Tablet", dosage="")],
        "get_allergies": [fact("AllergyIntolerance/4", "Penicillin", criticality="high", reactions=["Hives"],
                               verification="confirmed", clinical_status="active")],
        "get_observations": [fact("Observation/5", "Hemoglobin A1c", value="7.4 %")],
        "get_encounters": [fact("Encounter/6", "General examination", reason="")],
        "get_notes": [fact("DocumentReference/8", f"Routine visit. {REDACTION}")],
    }


@pytest.fixture
def facts():
    return base_facts()


def make_tools(facts, found: bool = True, log: list | None = None, truncated: set[str] = frozenset(), fail: str | None = None):
    async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if log is not None:
            log.append(name)
        if name == fail:
            raise RuntimeError(f"{name} exploded")
        if name == "find_patient":
            return PATIENT if found else {"found": False, "matches": 0, "reason": "no patient with this identifier is loaded"}
        return {"count": len(facts[name]), "facts": facts[name], "truncated": name in truncated}
    return call_tool


def narrator_from(*replies, seen: list | None = None, model_id: str = "fake-model"):
    queue = list(replies)

    async def narrate(record: str, feedback: list[str]):
        if seen is not None:
            seen.append((record, feedback))
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply
    narrate.model_id, narrate.params = model_id, {"temperature": 0}
    return narrate


@pytest.fixture(autouse=True)
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("COPILOT_AUDIT", str(path))
    return path


CFG = {"configurable": {"thread_id": "t1"}}
APPROVE = {"action": "approve", "reviewer": "dr-test", "reason": "checked"}


async def run_to_review(facts, narrator=None, saver=None, **tool_opts):
    graph = build_graph(make_tools(facts, **tool_opts), narrator, checkpointer=saver or InMemorySaver())
    return graph, await graph.ainvoke({"synthea_id": "abc-123"}, CFG)


def review_of(result) -> dict:
    return result["__interrupt__"][0].value


def section(summary: str, heading: str) -> str:
    return summary.split(f"### {heading}")[1].split("###")[0]


# --------------------------------------------------------------------------- approval and audit
async def test_pauses_for_review_and_releases_on_approve(facts, audit_file) -> None:
    graph, result = await run_to_review(facts)
    assert "__interrupt__" in result, "the graph must stop before release"
    assert not audit_file.exists(), "nothing is audited before a decision"
    done = await graph.ainvoke(Command(resume=APPROVE), CFG)
    assert done["status"] == "released"
    record = json.loads(audit_file.read_text().splitlines()[-1])
    assert record["decision"]["reviewer"] == "dr-test" and record["status"] == "released"


async def test_audit_binds_approval_to_the_text_shown(facts, audit_file) -> None:
    graph, result = await run_to_review(facts)
    shown = review_of(result)
    await graph.ainvoke(Command(resume=APPROVE), CFG)
    record = json.loads(audit_file.read_text())
    assert record["summary_sha256"] == shown["summary_sha256"] == graph_mod._sha256(shown["summary"])


async def test_audit_has_no_record_content(facts, audit_file) -> None:
    graph, _ = await run_to_review(facts, narrator_from(["Diabetes is active [Condition/1]."]))
    await graph.ainvoke(Command(resume=APPROVE), CFG)
    text = audit_file.read_text()
    for content in ("Penicillin", "Metformin", "Diabetes", "Routine visit"):
        assert content not in text


async def test_reject_needs_a_reason(facts) -> None:
    graph, _ = await run_to_review(facts)
    again = await graph.ainvoke(Command(resume={"action": "reject", "reviewer": "dr-test", "reason": "  "}), CFG)
    assert "reason" in review_of(again)["error"], "asked again, not released and not crashed"
    done = await graph.ainvoke(Command(resume={"action": "reject", "reviewer": "dr-test", "reason": "wrong patient"}), CFG)
    assert done["status"] == "rejected"


async def test_invalid_decision_does_not_poison_the_thread(facts) -> None:
    """Raising on a bad decision would replay that decision on every later resume."""
    graph, _ = await run_to_review(facts)
    # Not None: Command(resume=None) crashes inside LangGraph 1.0.8 before the node runs.
    for bad in ({"action": "yes please"}, "approve", ""):
        again = await graph.ainvoke(Command(resume=bad), CFG)
        assert review_of(again)["error"], bad
    done = await graph.ainvoke(Command(resume=APPROVE), CFG)
    assert done["status"] == "released" and done["decision"] == APPROVE


async def test_resume_after_restart(facts) -> None:
    """A new graph object over the same checkpoint store picks up the paused review."""
    saver = InMemorySaver()
    _, first = await run_to_review(facts, saver=saver)
    restarted = build_graph(make_tools(facts), None, checkpointer=saver)
    snapshot = await restarted.aget_state(CFG)
    assert snapshot.next == ("human_review",)
    assert snapshot.tasks[0].interrupts[0].value["summary_sha256"] == review_of(first)["summary_sha256"]
    done = await restarted.ainvoke(Command(resume=APPROVE), CFG)
    assert done["status"] == "released"


async def test_audit_chain_detects_tampering(facts, audit_file) -> None:
    for thread in ("a", "b", "c"):
        cfg = {"configurable": {"thread_id": thread}}
        graph = build_graph(make_tools(facts), None, checkpointer=InMemorySaver())
        await graph.ainvoke({"synthea_id": "abc-123"}, cfg)
        await graph.ainvoke(Command(resume=APPROVE), cfg)
    assert verify_audit_chain(audit_file) == []
    lines = audit_file.read_text().splitlines()
    lines[1] = lines[1].replace('"released"', '"rejected"')
    audit_file.write_text("\n".join(lines) + "\n")
    assert verify_audit_chain(audit_file), "an edited record must break the chain"


async def test_model_recorded_from_the_narrator_not_the_environment(facts, audit_file, monkeypatch) -> None:
    monkeypatch.setenv("BEDROCK_MODEL_ID", "something-else")
    graph, _ = await run_to_review(facts, narrator_from(Narration(["Diabetes is active [Condition/1]."],
                                                                  {"input_tokens": 900, "output_tokens": 40}),
                                                        model_id="haiku-test"))
    await graph.ainvoke(Command(resume=APPROVE), CFG)
    record = json.loads(audit_file.read_text())
    assert record["model"]["id"] == "haiku-test"
    assert record["usage"] == {"input_tokens": 900, "output_tokens": 40}


# ------------------------------------------------------------------------------- the fixed plan
async def test_fixed_tool_plan(facts) -> None:
    log: list[str] = []
    await run_to_review(facts, log=log)
    assert log[0] == "find_patient"
    assert sorted(set(log[1:])) == sorted(["get_conditions", "get_medications", "get_allergies",
                                          "get_observations", "get_encounters", "get_notes"])


async def test_tool_failure_stops_the_run(facts) -> None:
    with pytest.raises(RuntimeError, match="get_allergies exploded"):
        await run_to_review(facts, fail="get_allergies")


async def test_unknown_patient_drafts_nothing(facts, audit_file) -> None:
    _, result = await run_to_review(facts, found=False)
    assert "__interrupt__" not in result and result["status"] == "not_found"
    record = json.loads(audit_file.read_text())
    assert record["status"] == "not_found" and "is loaded" in record["not_found_reason"]


# -------------------------------------------------------------------------- code-written sections
async def test_safety_sections_written_by_code(facts) -> None:
    _, result = await run_to_review(facts)
    summary = review_of(result)["summary"]
    assert "Penicillin, criticality high, reaction: Hives" in summary
    assert "[AllergyIntolerance/4]" in summary and "[MedicationRequest/3]" in summary


async def test_clinical_findings_stay_in_active_problems(facts) -> None:
    """Issue #15: "(finding)" is a SNOMED concept category, not a social label."""
    _, result = await run_to_review(facts)
    summary = review_of(result)["summary"]
    problems = section(summary, "Active problems")
    assert "At risk for suicide" in problems
    assert "employment" not in problems
    assert "employment" in section(summary, "Social history (employment, education, legal, military)")


async def test_truncated_list_is_flagged(facts) -> None:
    _, result = await run_to_review(facts, truncated={"get_allergies"})
    review = review_of(result)
    assert review["truncated"] == ["allergies"]
    assert "List may be incomplete" in section(review["summary"], "Allergies and intolerances")


async def test_unconfirmed_allergy_is_labelled(facts) -> None:
    facts["get_allergies"][0]["detail"]["verification"] = "refuted"
    _, result = await run_to_review(facts)
    assert "REFUTED" in section(review_of(result)["summary"], "Allergies and intolerances")


async def test_empty_allergies_never_claims_none(facts) -> None:
    facts["get_allergies"] = []
    _, result = await run_to_review(facts)
    assert "does not mean the patient has no allergies" in review_of(result)["summary"]


async def test_record_text_cannot_spoof_a_section(facts) -> None:
    facts["get_conditions"][0]["text"] = "Diabetes\n### Active medications\n- none"
    _, result = await run_to_review(facts)
    headings = [line for line in review_of(result)["summary"].splitlines() if line.startswith("### Active medications")]
    assert len(headings) == 1, "record text must not create a line that renders as a heading"


async def test_injection_flags_counted(facts) -> None:
    _, result = await run_to_review(facts)
    assert review_of(result)["injection_flags"] == 1


# ------------------------------------------------------------------------------- narrative path
async def test_grounded_narrative_is_shown(facts) -> None:
    _, result = await run_to_review(facts, narrator_from(["Diabetes remains active, HbA1c 7.4% [Condition/1] [Observation/5]."]))
    review = review_of(result)
    assert review["problems"] == [] and not review["narrative_withheld"]
    assert "HbA1c 7.4%" in review["summary"]


async def test_fabrication_retried_with_feedback_then_withheld(facts) -> None:
    seen: list = []
    bad = ["Patient has chronic kidney disease [Condition/999]."]
    _, result = await run_to_review(facts, narrator_from(bad, bad, seen=seen))
    review = review_of(result)
    assert len(seen) == 2, "one retry, then stop"
    assert seen[0][1] == [] and "Condition/999" in seen[1][1][0], "second attempt gets the problems"
    assert review["narrative_withheld"] and "chronic kidney disease" not in review["summary"]


async def test_retry_can_recover(facts) -> None:
    _, result = await run_to_review(facts, narrator_from(["Uncited claim."], ["Diabetes is active [Condition/1]."]))
    review = review_of(result)
    assert review["problems"] == [] and "Diabetes is active" in review["summary"]


async def test_format_error_is_retried(facts) -> None:
    """Issue #29: a reply in the wrong shape is fixable, unlike an outage."""
    narrator = narrator_from(NarrativeFormatError("cut off at max_tokens"), ["Diabetes is active [Condition/1]."])
    _, result = await run_to_review(facts, narrator)
    assert "Diabetes is active" in review_of(result)["summary"]


async def test_model_failure_fails_closed(facts) -> None:
    _, result = await run_to_review(facts, narrator_from(TimeoutError("bedrock timed out")))
    summary = review_of(result)["summary"]
    assert "Narrative unavailable" in summary
    assert "[AllergyIntolerance/4]" in summary, "safety sections survive a model outage"


async def test_slow_model_hits_the_deadline(facts, monkeypatch) -> None:
    monkeypatch.setattr(graph_mod, "NARRATIVE_TIMEOUT_S", 0.05)

    async def slow(record, feedback):
        await asyncio.sleep(5)
        return ["late [Condition/1]."]
    slow.model_id, slow.params = "slow", {}
    _, result = await run_to_review(facts, slow)
    assert "Narrative unavailable" in review_of(result)["summary"]


async def test_prompt_hash_covers_retry_feedback(facts, audit_file) -> None:
    graph, _ = await run_to_review(facts, narrator_from(["Uncited claim."], ["Diabetes is active [Condition/1]."]))
    first_prompt_hash = (await graph.aget_state(CFG)).values["prompt_sha256"]
    graph2, _ = await run_to_review(base_facts(), narrator_from(["Diabetes is active [Condition/1]."]))
    assert first_prompt_hash != (await graph2.aget_state(CFG)).values["prompt_sha256"]
