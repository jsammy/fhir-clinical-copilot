"""Load Synthea FHIR R4 bundles into a FHIR server.

Order matters: patient bundles reference Organization, Location and Practitioner by identifier
(conditional references, e.g. "Organization?identifier=...|<uuid>"). The server resolves those
inside the transaction, so the hospital and practitioner bundles must be loaded first or every
patient bundle fails with "no matches for conditional reference".

Each patient file is a *transaction* bundle and is POSTed whole to the server root — never PUT
resource-by-resource — so a patient's record loads atomically or not at all.

Usage:
    python -m src.ingest.load_bundles --limit 50            # dev subset, seeded sample
    python -m src.ingest.load_bundles --limit 0             # everything (slow on H2)
    python -m src.ingest.load_bundles --skip-existing       # resume an interrupted load
    python -m src.ingest.load_bundles --base-url https://<healthlake-endpoint>/r4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

from src.ingest.saved_queries import SYNTHEA_ID_SYSTEM

SHARED_PREFIXES = ("hospitalInformation", "practitionerInformation")
FHIR_JSON = "application/fhir+json"


def split_files(fhir_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (shared reference bundles in load order, patient bundles)."""
    files = sorted(fhir_dir.glob("*.json"))
    shared = [f for p in SHARED_PREFIXES for f in files if f.name.startswith(p)]
    patients = [f for f in files if f not in shared]
    if len(shared) != len(SHARED_PREFIXES):
        sys.exit(f"Expected {SHARED_PREFIXES} bundles in {fhir_dir}, found {[f.name for f in shared]}")
    return shared, patients


def choose_patients(patients: list[Path], limit: int, max_mb: float, seed: int) -> list[Path]:
    """Seeded random sample rather than smallest-first: small bundles skew towards young
    patients with short histories, which would distort the age subgroup checks later."""
    eligible = [p for p in patients if p.stat().st_size <= max_mb * 1024 * 1024]
    if limit <= 0 or limit >= len(eligible):
        return eligible
    return sorted(random.Random(seed).sample(eligible, limit))


def post_bundle(client: httpx.Client, path: Path, retries: int = 3) -> tuple[bool, str]:
    body = path.read_bytes()
    for attempt in range(1, retries + 1):
        try:
            resp = client.post("", content=body, headers={"Content-Type": FHIR_JSON})
        except httpx.TransportError as exc:
            detail = f"transport error: {exc}"
        else:
            if resp.is_success:
                return True, f"{len(resp.json().get('entry', []))} entries"
            if resp.status_code < 500:
                # 4xx is a data/reference problem; retrying will not fix it.
                return False, f"HTTP {resp.status_code}: {_outcome_text(resp)}"
            detail = f"HTTP {resp.status_code}"
        time.sleep(2**attempt)
    return False, detail


def _outcome_text(resp: httpx.Response) -> str:
    try:
        issues = resp.json().get("issue", [])
        return "; ".join(i.get("diagnostics", "") for i in issues)[:300]
    except ValueError:
        return resp.text[:300]


def patient_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def already_loaded(client: httpx.Client, synthea_id: str) -> bool:
    """POST is not idempotent, so a re-run would duplicate every patient; check the identifier first."""
    resp = client.get(f"Patient?identifier={SYNTHEA_ID_SYSTEM}%7C{synthea_id}&_summary=count")
    resp.raise_for_status()
    return resp.json().get("total", 0) > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/fhir")
    ap.add_argument("--fhir-dir", type=Path, default=Path("data/fhir"))
    ap.add_argument("--limit", type=int, default=50, help="patients to load; 0 = all")
    ap.add_argument("--max-mb", type=float, default=10.0, help="skip patient bundles larger than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-shared", action="store_true", help="hospital/practitioner already loaded")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip patients already on the server (implies --skip-shared); resumes a broken load")
    args = ap.parse_args()

    shared, patients = split_files(args.fhir_dir)
    selected = choose_patients(patients, args.limit, args.max_mb, args.seed)
    print(f"{len(patients)} patient bundles found, loading {len(selected)} (<= {args.max_mb} MB, seed {args.seed})")

    loaded: list[str] = []
    failed: list[tuple[str, str]] = []
    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        client.get("/metadata").raise_for_status()

        if not (args.skip_shared or args.skip_existing):
            for path in shared:
                ok, detail = post_bundle(client, path)
                print(f"[shared] {path.name}: {'ok' if ok else 'FAILED'} ({detail})")
                if not ok:
                    sys.exit("Shared reference bundles must load before patients; stopping.")

        for i, path in enumerate(selected, 1):
            if args.skip_existing and already_loaded(client, patient_id(path)):
                print(f"[{i}/{len(selected)}] {path.name[:60]}: already on server, skipped")
                loaded.append(patient_id(path))
                continue
            start = time.perf_counter()
            ok, detail = post_bundle(client, path)
            secs = time.perf_counter() - start
            print(f"[{i}/{len(selected)}] {path.name[:60]}: {'ok' if ok else 'FAILED'} ({detail}, {secs:.1f}s)")
            if ok:
                loaded.append(patient_id(path))
            else:
                failed.append((path.name, detail))

    # Manifest of loaded patients, as Synthea UUIDs (business identifiers, also the CSV Id column),
    # not server logical ids — those differ per server. Lets eval and CSV checks share the cohort.
    manifest = args.fhir_dir.parent / "loaded_patients.json"
    manifest.write_text(json.dumps({"base_url": args.base_url, "seed": args.seed, "patient_ids": loaded}, indent=2))
    print(f"\nLoaded {len(loaded)}, failed {len(failed)}. Manifest: {manifest}")
    for name, detail in failed:
        print(f"  FAILED {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
