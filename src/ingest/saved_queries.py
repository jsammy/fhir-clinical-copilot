"""Ten saved FHIR R4 searches: a smoke test of the load and a reference for the search syntax
the retrieval layer and MCP tools will use.

Each query maps to a question a clinician would ask, and the comment names the FHIR search
feature it exercises. Patient-scoped queries use the first patient in data/loaded_patients.json
unless --patient is given.

Two IDs exist for every patient. The Synthea UUID is a *business identifier* (system|value):
stable, the same in the CSV extract and on any server. The *logical id* (Patient/2507) is
assigned by whichever server stored the record, because the bundles POST rather than PUT. We keep
the portable one and resolve the server's logical id at query time.

Usage:
    python -m src.ingest.saved_queries
    python -m src.ingest.saved_queries --patient <synthea-uuid> --show 3   # print query 3's JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

LOINC_HBA1C = "http://loinc.org|4548-4"
SNOMED_DIABETES = "http://snomed.info/sct|44054006"
SYNTHEA_ID_SYSTEM = "https://github.com/synthetichealth/synthea"

# (label, path) — "{pid}" is filled with the patient id.
QUERIES: list[tuple[str, str]] = [
    # 1. Capability statement: which resources and search params this server supports.
    ("Server capabilities", "metadata"),
    # 2. Plain search with sort and paging (_count); the response is a searchset Bundle.
    ("Five oldest patients", "Patient?_sort=birthdate&_count=5"),
    # 3. $everything operation: the whole patient compartment — what retrieval will flatten.
    ("Full record ($everything)", "Patient/{pid}/$everything?_count=50"),
    # 4. Token search on a coded element: the active problem list.
    ("Active problem list", "Condition?patient={pid}&clinical-status=active"),
    # 5. _include pulls the referenced Medication alongside each order, if it is not inline.
    ("Active medications", "MedicationRequest?patient={pid}&status=active&_include=MedicationRequest:medication"),
    # 6. Allergies. An empty result means "none recorded", not "no allergies" — the copilot must
    #    say exactly that, never "patient has no allergies".
    ("Allergies", "AllergyIntolerance?patient={pid}"),
    # 7. system|code token plus newest-first sort: an HbA1c trend.
    ("Latest HbA1c results", f"Observation?patient={{pid}}&code={LOINC_HBA1C}&_sort=-date&_count=5"),
    # 8. Category filter plus a date prefix (ge = on or after).
    ("Vital signs since 2020", "Observation?patient={pid}&category=vital-signs&date=ge2020-01-01&_count=20"),
    # 9. _revinclude: the latest encounters plus the conditions recorded during them.
    ("Recent encounters + diagnoses", "Encounter?patient={pid}&_sort=-date&_count=3&_revinclude=Condition:encounter"),
    # 10. _has reverse chaining across the cohort, with _summary=count (a total, no payload).
    ("Cohort: patients with diabetes", f"Patient?_has:Condition:patient:code={SNOMED_DIABETES}&_summary=count"),
]


def default_patient(manifest: Path) -> str:
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found: run load_bundles first or pass --patient")
    ids = json.loads(manifest.read_text())["patient_ids"]
    if not ids:
        raise SystemExit(f"{manifest} lists no loaded patients")
    return ids[0]


def resolve_logical_id(client: httpx.Client, synthea_id: str) -> str:
    """Business identifier -> this server's logical id."""
    resp = client.get(encode_pipes(f"Patient?identifier={SYNTHEA_ID_SYSTEM}|{synthea_id}&_elements=id"))
    resp.raise_for_status()
    entries = resp.json().get("entry", [])
    if len(entries) != 1:
        # 0 = not loaded here; >1 = the same bundle was loaded twice (POST is not idempotent).
        raise SystemExit(f"Expected 1 Patient with identifier {synthea_id}, found {len(entries)}")
    return entries[0]["resource"]["id"]


def encode_pipes(path: str) -> str:
    # system|code tokens: Tomcat (under HAPI) rejects a raw "|" in the URL with HTTP 400.
    return path.replace("|", "%7C")


def summarise(body: dict) -> str:
    if body.get("resourceType") != "Bundle":
        return body.get("resourceType", "?")
    entries = body.get("entry", [])
    types: dict[str, int] = {}
    for e in entries:
        rt = e["resource"]["resourceType"]
        types[rt] = types.get(rt, 0) + 1
    total = body.get("total")
    by_type = ", ".join(f"{k} {v}" for k, v in sorted(types.items()))
    return f"total={total if total is not None else 'n/a'} page={len(entries)} [{by_type}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/fhir")
    ap.add_argument("--patient", help="Synthea UUID; defaults to the first loaded patient")
    ap.add_argument("--logical-id", help="use this server id directly (e.g. on the public HAPI server)")
    ap.add_argument("--manifest", type=Path, default=Path("data/loaded_patients.json"))
    ap.add_argument("--show", type=int, help="print the full JSON response of query N (1-10)")
    args = ap.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=60.0, headers={"Accept": "application/fhir+json"}) as client:
        if args.logical_id:
            pid = args.logical_id
            print(f"Patient: {pid} (logical id)\n")
        else:
            synthea_id = args.patient or default_patient(args.manifest)
            pid = resolve_logical_id(client, synthea_id)
            print(f"Patient: {synthea_id} (Synthea) -> Patient/{pid} on this server\n")
        for n, (label, template) in enumerate(QUERIES, 1):
            path = template.format(pid=pid)
            resp = client.get(encode_pipes(path))
            status = summarise(resp.json()) if resp.is_success else f"HTTP {resp.status_code}"
            print(f"{n:>2}. {label:<32} {status}\n    GET {path}")
            if args.show == n:
                print(json.dumps(resp.json(), indent=2)[:20000])


if __name__ == "__main__":
    main()
