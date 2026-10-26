"""Semantic + keyword search over clinical notes, one patient at a time.

Coded facts (medications, labs, diagnoses) are answered by FHIR queries, which are exact and
current. This index exists for what FHIR search cannot do: find meaning in free-text notes
("has he mentioned breathing problems?"). It backs the `search_notes` MCP tool.

Chunking: every Synthea note has the same sections (Chief Complaint, History of Present Illness,
Social History, Allergies, Medications, Assessment and Plan). One chunk = one section of one note.
Sections repeat from visit to visit, so identical text is stored once per patient and cites the
most recent note it appears in.

Hybrid search: keyword (BM25) and vector (cosine) rankings are fused with Reciprocal Rank Fusion.
Keyword search catches exact terms (drug names, "lisinopril"); vectors catch paraphrase. Both run
only over the requested patient's chunks: the patient filter is applied before any scoring, and
it comes from the caller's patient id, never from the query text.

    python -m src.retrieval.notes_index build            # all loaded patients, NOTES_EMBEDDER=hash|bedrock
    python -m src.retrieval.notes_index search 19764 "breathing problems"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.retrieval.embedders import Embedder, make_embedder, tokenize

INDEX_DIR = Path("data/index/notes")
SECTION_RE = re.compile(r"^#{1,2}\s+(.+?)\s*$", re.MULTILINE)
RRF_K = 60  # standard Reciprocal Rank Fusion constant: damps the weight of top ranks
MIN_CHUNK_CHARS = 20  # "No complaints." style sections carry nothing worth retrieving


@dataclass
class Chunk:
    patient_ref: str  # "Patient/19764": the filter key
    ref: str          # "DocumentReference/20161": the citation target (most recent occurrence)
    date: str | None
    section: str
    text: str
    occurrences: int = 1  # how many notes contain this exact section text


def split_note(text: str) -> list[tuple[str, str]]:
    """[(section heading, body)] from a Synthea-style note; a note without headings is one chunk."""
    marks = list(SECTION_RE.finditer(text))
    if not marks:
        return [("Note", text.strip())] if text.strip() else []
    parts = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = text[m.end(): nxt.start() if nxt else len(text)].strip()
        parts.append((m.group(1), " ".join(body.split())))
    return parts


def chunk_notes(patient_ref: str, notes: list[dict]) -> list[Chunk]:
    """notes: [{"ref", "date", "text"}], any order. Identical (section, text) is stored once."""
    by_key: dict[tuple[str, str], Chunk] = {}
    for note in sorted(notes, key=lambda n: n.get("date") or "", reverse=True):  # newest first
        for section, body in split_note(note["text"]):
            if len(body) < MIN_CHUNK_CHARS:
                continue
            key = (section, body)
            if key in by_key:
                by_key[key].occurrences += 1
            else:
                by_key[key] = Chunk(patient_ref, note["ref"], note.get("date"), section, body)
    return list(by_key.values())


class NotesIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        self.chunks, self.vectors, self.embedder_name = chunks, vectors, embedder_name
        self._by_patient: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self._by_patient.setdefault(c.patient_ref, []).append(i)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> NotesIndex:
        texts = [f"{c.section}: {c.text}" for c in chunks]
        vectors = embedder.embed(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(chunks, vectors, embedder.name)

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c)) + "\n")
        (directory / "meta.json").write_text(json.dumps({"embedder": self.embedder_name, "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> NotesIndex:
        meta = json.loads((directory / "meta.json").read_text())
        chunks = [Chunk(**json.loads(line)) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embedder"])

    def search(self, patient_ref: str, query: str, embedder: Embedder, k: int = 5) -> list[dict]:
        if embedder.name != self.embedder_name:
            raise ValueError(f"index built with {self.embedder_name}, query embedder is {embedder.name}")
        ids = self._by_patient.get(patient_ref, [])  # the patient filter comes first
        if not ids or not query.strip():
            return []
        keyword = _bm25_rank(query, [self.chunks[i] for i in ids])
        sims = self.vectors[ids] @ embedder.embed([query])[0]
        vector = [j for j in np.argsort(-sims) if sims[j] > 0]
        fused: dict[int, float] = {}
        found_by: dict[int, set[str]] = {}
        for name, ranking in (("keyword", keyword), ("vector", vector)):
            for rank, j in enumerate(ranking[: max(20, k * 4)]):
                fused[j] = fused.get(j, 0.0) + 1.0 / (RRF_K + rank + 1)
                found_by.setdefault(j, set()).add(name)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**asdict(self.chunks[ids[j]]), "score": round(fused[j], 5),
                 "matched_by": sorted(found_by[j])} for j in best]


def _bm25_rank(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of chunks with any query term, best first. Computed over this patient's chunks only."""
    docs = [tokenize(f"{c.section} {c.text}") for c in chunks]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    terms = set(tokenize(query))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1)
                / (tf[t] + k1 * (1 - b + b * len(d) / avg)) for t in terms if t in tf)
        if s > 0:
            scores.append((s, i))
    return [i for _, i in sorted(scores, reverse=True)]


# -------------------------------------------------------------------------------------- build CLI
def build_from_fhir(base_url: str, manifest: Path, max_notes: int = 500) -> list[Chunk]:
    from src.agent.fhir_read import ReadOnlyFhir

    fhir = ReadOnlyFhir(base_url)
    chunks: list[Chunk] = []
    for synthea_id in json.loads(manifest.read_text())["patient_ids"]:
        patient, _ = fhir.find_patient(synthea_id)
        if patient is None:
            continue
        notes = fhir.notes(patient["logical_id"], count=max_notes, max_chars=20000).facts
        chunks += chunk_notes(patient["ref"], [{"ref": n.ref, "date": n.date, "text": n.text} for n in notes])
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base-url", default="http://localhost:8080/fhir")
    b.add_argument("--manifest", type=Path, default=Path("data/loaded_patients.json"))
    s = sub.add_parser("search")
    s.add_argument("patient_id", help="logical id on the FHIR server, e.g. 19764")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    embedder = make_embedder()
    if args.cmd == "build":
        chunks = build_from_fhir(args.base_url, args.manifest)
        index = NotesIndex.build(chunks, embedder)
        index.save()
        print(f"{len(chunks)} chunks from {len(index._by_patient)} patients, embedder {embedder.name} -> {INDEX_DIR}")
    else:
        for hit in NotesIndex.load().search(f"Patient/{args.patient_id}", args.query, embedder, args.k):
            print(f"{hit['score']:.4f} {','.join(hit['matched_by']):14} [{hit['ref']}] {hit['date']} "
                  f"{hit['section']} (x{hit['occurrences']}): {hit['text'][:110]}")


if __name__ == "__main__":
    main()
