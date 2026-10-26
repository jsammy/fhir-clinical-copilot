"""Chunking, hybrid search and the patient filter, with the offline embedder."""

import pytest

from src.retrieval.embedders import HashEmbedder, tokenize
from src.retrieval.notes_index import NotesIndex, chunk_notes, split_note

NOTE = """2021-07-17

# Chief Complaint
No complaints.

# History of Present Illness
Leif534 is a 27 year-old male with perennial allergic rhinitis.

# Social History
Patient is married. Patient is an active smoker and is an alcoholic.

# Assessment and Plan
Patient is presenting with viral sinusitis (disorder).
"""


def note(ref, date, text=NOTE):
    return {"ref": ref, "date": date, "text": text}


def test_split_note_by_section() -> None:
    sections = dict(split_note(NOTE))
    assert sections["Social History"].startswith("Patient is married")
    assert set(sections) == {"Chief Complaint", "History of Present Illness", "Social History", "Assessment and Plan"}


def test_repeated_sections_stored_once_citing_the_newest_note() -> None:
    chunks = chunk_notes("Patient/1", [note("DocumentReference/1", "2019-01-01"), note("DocumentReference/2", "2021-07-17")])
    social = [c for c in chunks if c.section == "Social History"]
    assert len(social) == 1 and social[0].ref == "DocumentReference/2" and social[0].occurrences == 2
    assert not any(c.section == "Chief Complaint" for c in chunks), "'No complaints.' is too short to index"


@pytest.fixture
def index():
    other = NOTE.replace("active smoker", "never smoked").replace("viral sinusitis", "asthma")
    chunks = chunk_notes("Patient/1", [note("DocumentReference/1", "2021-07-17")]) \
        + chunk_notes("Patient/2", [note("DocumentReference/9", "2021-07-17", other)])
    return NotesIndex.build(chunks, HashEmbedder())


def test_search_never_crosses_patients(index) -> None:
    hits = index.search("Patient/2", "active smoker", HashEmbedder(), k=10)
    assert hits and all(h["patient_ref"] == "Patient/2" for h in hits)
    assert all("DocumentReference/1" != h["ref"] for h in hits)


def test_unknown_patient_returns_nothing(index) -> None:
    assert index.search("Patient/404", "smoker", HashEmbedder()) == []


def test_keyword_search_ignores_stopwords_and_stems(index) -> None:
    # Regression: "and" once matched every "Assessment and Plan" section.
    top = index.search("Patient/1", "smoking and alcohol", HashEmbedder(), k=1)[0]
    assert top["section"] == "Social History"
    assert "and" not in tokenize("smoking and alcohol") and tokenize("smoker") == tokenize("smoking")


def test_hits_explain_how_they_matched(index) -> None:
    hit = index.search("Patient/1", "sinusitis", HashEmbedder(), k=1)[0]
    assert hit["section"] == "Assessment and Plan" and "keyword" in hit["matched_by"]


def test_index_refuses_a_different_embedder(index) -> None:
    with pytest.raises(ValueError, match="built with"):
        index.search("Patient/1", "smoker", HashEmbedder(dim=256))


def test_save_and_load_round_trip(index, tmp_path) -> None:
    index.save(tmp_path)
    loaded = NotesIndex.load(tmp_path)
    assert loaded.search("Patient/1", "smoker", HashEmbedder(), k=1) == index.search("Patient/1", "smoker", HashEmbedder(), k=1)
