import json

from src.eval.routing_eval import GOLDEN, score


def result(tools, status="answered", **flags):
    return {"status": status, "tool_log": [{"tool": t, "outcome": "ok", **flags} for t in tools]}


def test_routing_scored_by_tool_kind_and_named_tool() -> None:
    case = {"id": "bp", "expect": "fhir", "tool": "get_observations"}
    assert score(case, result(["get_observations"]))["routed"]
    assert not score(case, result(["get_medications"]))["routed"], "right kind, wrong tool"
    assert not score(case, result(["search_notes"]))["routed"]


def test_none_cases_must_say_not_in_record() -> None:
    case = {"id": "x", "expect": "none"}
    assert score(case, result(["search_notes"], status="not_in_record"))["routed"]
    assert not score(case, result(["search_notes"], status="answered"))["routed"]


def test_bounded_fails_when_the_model_tried_another_patient() -> None:
    assert not score({"id": "x", "expect": "fhir"}, result(["get_medications"], patient_id_ignored=True))["bounded"]


def test_golden_set_is_well_formed() -> None:
    cases = json.loads(GOLDEN.read_text())["cases"]
    assert len({c["id"] for c in cases}) == len(cases) >= 10
    assert {c["expect"] for c in cases} == {"fhir", "notes", "none"}
