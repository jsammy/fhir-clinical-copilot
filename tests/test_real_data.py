"""Guards checked against real Synthea data, not hand-picked examples. Every issue in the first
code review that unit tests missed was found this way.

- vocabulary tests need data/csv/conditions.csv (gitignored; skipped if absent);
- `live` tests need the local HAPI server with the 50-patient cohort: `pytest -m live`.
"""

import csv
import json
from pathlib import Path

import httpx
import pytest

from src.agent.fhir_read import ReadOnlyFhir
from src.agent.graph import SOCIAL_HISTORY_SNOMED, render_sections
from src.agent.guards import REDACTION

CONDITIONS_CSV = Path("data/csv/conditions.csv")
MANIFEST = Path("data/loaded_patients.json")
HAPI = "http://localhost:8080/fhir"
INJECTION_FIXTURE_PATIENT = "ce6fbb19-19f4-b597-7df8-8bb192f78da8"  # carries the planted test note

PATIENT = {"ref": "Patient/1", "name": "Vocabulary Check", "gender": "female", "birth_date": "1970-01-01"}


@pytest.mark.skipif(not CONDITIONS_CSV.exists(), reason="Synthea CSV extract not present")
def test_no_clinical_concept_leaves_the_problem_list() -> None:
    """Issue #15: render every distinct Synthea condition; only the explicit social codes may move."""
    concepts = {}
    with CONDITIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            concepts[row["CODE"]] = row["DESCRIPTION"]
    conditions = [{"ref": f"Condition/{i}", "kind": "Condition", "text": text, "date": None, "detail": {"snomed": code}}
                  for i, (code, text) in enumerate(concepts.items())]
    empty = {k: [] for k in ("medications", "allergies", "labs", "vitals", "encounters", "notes")}
    summary = render_sections(PATIENT, {**empty, "conditions": conditions})
    problems = summary.split("### Active problems")[1].split("###")[0]
    for c in conditions:
        expected_social = c["detail"]["snomed"] in SOCIAL_HISTORY_SNOMED
        assert (f"[{c['ref']}]" in problems) != expected_social, f"{c['text']} is in the wrong section"
    assert "At risk for suicide" in problems


def _hapi_up() -> bool:
    try:
        return httpx.get(f"{HAPI}/metadata", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.live
@pytest.mark.skipif(not MANIFEST.exists(), reason="no loaded-patient manifest")
def test_injection_filter_leaves_real_notes_alone() -> None:
    """Issue #8: a filter that edits clinical text is a defect. Real Synthea notes contain no
    attacks, so any redaction in them is a false positive."""
    if not _hapi_up():
        pytest.skip("HAPI not running")
    fhir = ReadOnlyFhir(HAPI)
    checked, false_positives = 0, []
    for synthea_id in json.loads(MANIFEST.read_text())["patient_ids"]:
        if synthea_id == INJECTION_FIXTURE_PATIENT:
            continue
        patient, _ = fhir.find_patient(synthea_id)
        for note in fhir.notes(patient["logical_id"], count=5).facts:
            checked += 1
            if REDACTION in note.text:
                false_positives.append(note.ref)
    assert checked > 100, "expected a few hundred notes across the cohort"
    assert false_positives == []


@pytest.mark.live
def test_planted_injection_is_caught() -> None:
    if not _hapi_up():
        pytest.skip("HAPI not running")
    fhir = ReadOnlyFhir(HAPI)
    patient, _ = fhir.find_patient(INJECTION_FIXTURE_PATIENT)
    texts = [n.text for n in fhir.notes(patient["logical_id"], count=5).facts]
    assert any(t.count(REDACTION) >= 3 for t in texts), "the planted note has three instruction spans"
