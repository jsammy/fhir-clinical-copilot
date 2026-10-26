"""Read-only FHIR access for the agent layer.

The safety property "the agent cannot change a record" is enforced here, below the MCP server,
not by asking the model nicely:

- the only HTTP verb this module can issue is GET (there is no post/put/delete code path);
- only allow-listed resource types can be queried, and only patient-scoped searches;
- every result is reduced to a compact `Fact` that keeps its FHIR reference ("Condition/123")
  so the summary can cite it, and every free-text field passes through the injection guard.

Two kinds of read, with different completeness rules:

- *complete lists* (conditions, medications, allergies) are followed page by page up to a hard cap.
  If the cap is hit, the result says `truncated=True`, because an incomplete safety list must never
  look complete;
- *latest N* reads (observations, encounters, notes) stop as soon as N results are in.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from pydantic import BaseModel

from src.agent.guards import check_id, sanitize_text
from src.ingest.saved_queries import SYNTHEA_ID_SYSTEM, encode_pipes

ALLOWED_RESOURCES = frozenset(
    {"Patient", "Condition", "MedicationRequest", "AllergyIntolerance", "Observation", "Encounter",
     "DocumentReference"}
)
MAX_PAGES = 5

# Clinical statuses that mean "this problem is current". Recurrence and relapse are active too.
ACTIVE_CONDITION_STATUSES = "active,recurrence,relapse"


class Fact(BaseModel):
    """One citable statement taken from one FHIR resource."""

    ref: str  # "Condition/123" — the citation target
    kind: str  # resource type
    text: str  # human-readable, sanitised
    date: str | None = None
    detail: dict[str, Any] = {}


class Facts(BaseModel):
    count: int
    facts: list[Fact]
    # True when a complete-list read hit the page cap: more records exist than were returned.
    truncated: bool = False


class ReadOnlyFhir:
    def __init__(self, base_url: str, timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        # `transport` lets tests replay recorded FHIR responses without a server.
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout, headers={"Accept": "application/fhir+json"}, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    # -- the only network call in this module ------------------------------------------------
    def _search(self, resource: str, params: str, limit: int | None = None
                ) -> tuple[list[dict], dict[str, dict], bool]:
        """(matches, {"Type/id": resource} for _include'd entries, truncated).

        limit=None reads a complete list (up to MAX_PAGES) and reports truncation; a limit stops as
        soon as that many matches are in, since "latest N" is complete by definition. Included
        entries are kept apart so a referenced Medication is never mistaken for a patient fact."""
        if resource not in ALLOWED_RESOURCES:
            raise PermissionError(f"{resource} is not on the read allow-list")
        url: str | None = encode_pipes(f"{resource}?{params}")
        matches: list[dict] = []
        includes: dict[str, dict] = {}
        for _ in range(MAX_PAGES):
            resp = self._client.get(url)
            resp.raise_for_status()
            body = resp.json()
            for e in body.get("entry", []):
                if e.get("search", {}).get("mode", "match") == "match":
                    matches.append(e["resource"])
                else:
                    includes[_ref(e["resource"])] = e["resource"]
            url = next((link["url"] for link in body.get("link", []) if link["relation"] == "next"), None)
            if url is None or (limit is not None and len(matches) >= limit):
                break
        truncated = limit is None and url is not None
        return (matches[:limit] if limit else matches), includes, truncated

    # -- patient resolution ------------------------------------------------------------------
    def find_patient(self, synthea_id: str) -> tuple[dict | None, int]:
        """(patient, number of matches). Only exactly one match resolves; the count says whether
        the patient is missing (0) or was loaded twice (2+)."""
        found, _, _ = self._search("Patient", f"identifier={SYNTHEA_ID_SYSTEM}|{check_id(synthea_id)}")
        if len(found) != 1:
            return None, len(found)
        p = found[0]
        name = (p.get("name") or [{}])[0]
        return {
            "ref": f"Patient/{p['id']}",
            "logical_id": p["id"],
            # Synthea appends digits to names ("Aimee901"); strip them for display only.
            "name": " ".join(_strip_digits(s) for s in name.get("given", []) + [name.get("family", "")]).strip(),
            "gender": p.get("gender"),
            "birth_date": p.get("birthDate"),
            "deceased": "deceasedDateTime" in p or p.get("deceasedBoolean", False),
            "deceased_date": p.get("deceasedDateTime", "")[:10] or None,
        }, 1

    # -- patient-scoped reads ----------------------------------------------------------------
    def conditions(self, pid: str, active_only: bool = True) -> Facts:
        check_id(pid)
        q = f"patient={pid}" + (f"&clinical-status={ACTIVE_CONDITION_STATUSES}" if active_only else "") + "&_count=100"
        rows, _, truncated = self._search("Condition", q)
        return _facts([
            Fact(ref=_ref(r), kind="Condition", text=_code_text(r.get("code")),
                 date=r.get("onsetDateTime", "")[:10] or None,
                 detail={"status": _first_code(r.get("clinicalStatus")),
                         "snomed": _first_code(r.get("code"), "http://snomed.info/sct")})
            for r in rows
        ], truncated)

    def medications(self, pid: str) -> Facts:
        check_id(pid)
        # Synthea names most drugs inline (medicationCodeableConcept) but some orders point at a
        # separate Medication resource; _include fetches those in the same request.
        rows, included, truncated = self._search(
            "MedicationRequest", f"patient={pid}&status=active&_include=MedicationRequest:medication&_count=100")

        def drug(r: dict) -> str:
            if text := _code_text(r.get("medicationCodeableConcept")):
                return text
            med = included.get(r.get("medicationReference", {}).get("reference", ""))
            return _code_text((med or {}).get("code")) or "medication (name not resolvable)"

        return _facts([
            Fact(ref=_ref(r), kind="MedicationRequest", text=drug(r),
                 date=r.get("authoredOn", "")[:10] or None,
                 detail={"dosage": sanitize_text(((r.get("dosageInstruction") or [{}])[0]).get("text", ""))})
            for r in rows
        ], truncated)

    def allergies(self, pid: str) -> Facts:
        check_id(pid)
        rows, _, truncated = self._search("AllergyIntolerance", f"patient={pid}&_count=100")
        facts = []
        for r in rows:
            verification = _first_code(r.get("verificationStatus"))
            if verification == "entered-in-error":
                continue  # recorded by mistake: not a fact about the patient
            facts.append(Fact(
                ref=_ref(r), kind="AllergyIntolerance", text=_code_text(r.get("code")),
                date=r.get("recordedDate", "")[:10] or None,
                detail={"criticality": r.get("criticality"),
                        "clinical_status": _first_code(r.get("clinicalStatus")),
                        "verification": verification,
                        "reactions": [_code_text(m) for rx in r.get("reaction", []) for m in rx.get("manifestation", [])]}))
        return _facts(facts, truncated)

    def observations(self, pid: str, category: str = "laboratory", count: int = 20) -> Facts:
        check_id(pid)
        rows, _, _ = self._search("Observation", f"patient={pid}&category={category}&_sort=-date&_count={count}", limit=count)
        return _facts([
            Fact(ref=_ref(r), kind="Observation", text=_code_text(r.get("code")),
                 date=r.get("effectiveDateTime", "")[:10] or None, detail={"value": _value(r)})
            for r in rows
        ])

    def encounters(self, pid: str, count: int = 5) -> Facts:
        check_id(pid)
        rows, _, _ = self._search("Encounter", f"patient={pid}&_sort=-date&_count={count}", limit=count)
        return _facts([
            Fact(ref=_ref(r), kind="Encounter",
                 text=_code_text((r.get("type") or [None])[0]) or "encounter",
                 date=r.get("period", {}).get("start", "")[:10] or None,
                 detail={"class": r.get("class", {}).get("code"),
                         "reason": ", ".join(_code_text(c) for c in r.get("reasonCode", []))})
            for r in rows
        ])

    def notes(self, pid: str, count: int = 2, max_chars: int = 1500) -> Facts:
        """Free-text clinical notes: the highest prompt-injection exposure in the record, because
        their content is authored text rather than coded data."""
        check_id(pid)
        rows, _, _ = self._search("DocumentReference", f"patient={pid}&_sort=-date&_count={count}", limit=count)
        facts = []
        for r in rows:
            att = ((r.get("content") or [{}])[0]).get("attachment", {})
            raw = base64.b64decode(att["data"]).decode("utf-8", "replace") if att.get("data") else ""
            facts.append(Fact(ref=_ref(r), kind="DocumentReference", text=sanitize_text(raw[:max_chars]),
                              date=r.get("date", "")[:10] or None))
        return _facts(facts)


def _facts(items: list[Fact], truncated: bool = False) -> Facts:
    return Facts(count=len(items), facts=items, truncated=truncated)


def _ref(r: dict) -> str:
    return f"{r['resourceType']}/{r['id']}"


def _code_text(cc: dict | None) -> str:
    if not cc:
        return ""
    text = cc.get("text") or next((c.get("display") for c in cc.get("coding", []) if c.get("display")), "")
    return sanitize_text(text)


def _first_code(cc: dict | None, system: str | None = None) -> str | None:
    return next((c.get("code") for c in (cc or {}).get("coding", []) if system is None or c.get("system") == system), None)


def _value(r: dict) -> str:
    if q := r.get("valueQuantity"):
        return f"{round(q.get('value', 0), 2)} {q.get('unit', '')}".strip()
    if cc := r.get("valueCodeableConcept"):
        return _code_text(cc)
    if comps := r.get("component"):  # e.g. blood pressure: systolic + diastolic
        return "; ".join(f"{_code_text(c.get('code'))} {_value(c)}" for c in comps)
    return sanitize_text(str(r.get("valueString", "")))


def _strip_digits(s: str) -> str:
    return "".join(ch for ch in s if not ch.isdigit())
