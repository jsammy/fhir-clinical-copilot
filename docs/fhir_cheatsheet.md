# FHIR R4 cheat-sheet

The subset of HL7 FHIR R4 this project uses, with the exact searches it runs. Every query below
was run against the local HAPI server (`src/ingest/saved_queries.py`) and the public test server
`https://hapi.fhir.org/baseR4`.

## Resources

| Resource | What it holds | Fields this project reads |
|---|---|---|
| **Patient** | Demographics, identifiers | `identifier` (Synthea UUID), `name`, `gender`, `birthDate`, `deceasedDateTime` |
| **Encounter** | One contact: visit, admission, telehealth | `type`, `class` (AMB/EMER/IMP), `period.start`, `reasonCode` |
| **Condition** | Problem-list entry or encounter diagnosis | `code` (SNOMED CT), `clinicalStatus` (active/resolved), `onsetDateTime` |
| **Observation** | A measurement: lab, vital sign, survey score | `code` (LOINC), `category`, `effectiveDateTime`, `valueQuantity` or `component[]` (BP) |
| **MedicationRequest** | A prescription or order | `status`, `medicationCodeableConcept` *or* `medicationReference`, `dosageInstruction[0].text` |
| **Medication** | The drug itself, when an order references it instead of naming it inline | `code` (RxNorm) |
| **AllergyIntolerance** | A recorded allergy | `code`, `criticality` (low/high), `reaction[].manifestation` |
| **DiagnosticReport** | A grouped result (lab panel, imaging report) | `result[]` → Observations |
| **DocumentReference** | A clinical document, e.g. a note | `date`, `content[0].attachment.data` (base64 text) |
| **Bundle** | A container of resources | `type`: `transaction` (load), `searchset` (results), `collection` |

## Two kinds of id

| | Example | Stable across servers? | Use it for |
|---|---|---|---|
| **Business identifier** | `identifier=https://github.com/synthetichealth/synthea\|1b18…` | Yes. Same in the CSV extract | Finding a patient anywhere, joining to the CSV |
| **Logical id** | `Patient/2507` | No. Assigned by the server on POST | Citations, references, follow-up reads on *this* server |

Synthea bundles are POSTed, so the server assigns logical ids. Resolve the business identifier first,
then use the logical id for everything else.

## Search syntax

| Feature | Example | Notes |
|---|---|---|
| Token (`system\|code`) | `Observation?code=http://loinc.org\|4548-4` | Send `\|` as `%7C`: Tomcat under HAPI rejects a raw `\|` |
| Token, code only | `Condition?clinical-status=active` | |
| Reference | `Condition?patient=2507` | Same as `subject=Patient/2507` |
| Date prefix | `Observation?date=ge2020-01-01` | `eq ne gt lt ge le sa eb` |
| Sort | `_sort=-date` | `-` = descending |
| Page size | `_count=50` | Server may cap it |
| Paging | follow `link[relation=next].url` | Never build page URLs by hand |
| Include referenced | `MedicationRequest?…&_include=MedicationRequest:medication` | Included entries have `search.mode = include` |
| Include referencing | `Encounter?…&_revinclude=Condition:encounter` | Pulls Conditions that point at each Encounter |
| Reverse chain | `Patient?_has:Condition:patient:code=…\|44054006` | Patients *with* a diabetes Condition |
| Count only | `_summary=count` | Returns `total`, no entries |
| Trim payload | `_elements=id,subject` | |
| Whole compartment | `Patient/2507/$everything` | Everything about one patient, paged |

## Bundles

- **transaction**: all entries succeed or none do. Synthea patient files are transaction bundles,
  POSTed whole to the server root. Their conditional references (`Organization?identifier=…`)
  resolve only if the shared hospital and practitioner bundles were loaded first.
- **searchset**: what a search returns. `total` may be absent; `entry[].search.mode` is `match`
  or `include`.
- POST is not idempotent: loading the same bundle twice creates duplicate patients. The loader's
  `--skip-existing` checks the identifier first.

## Codes you meet in Synthea

| System | URL | Example |
|---|---|---|
| SNOMED CT | `http://snomed.info/sct` | `44054006` type 2 diabetes |
| LOINC | `http://loinc.org` | `4548-4` HbA1c, `85354-9` blood pressure panel |
| RxNorm | `http://www.nlm.nih.gov/research/umls/rxnorm` | `860975` metformin 500 mg |

Synthea also records social circumstances (employment, education, housing) as Conditions with
the SNOMED semantic tag `(finding)`. The agent lists those separately from clinical problems.

## Pitfalls seen in this project

- An empty AllergyIntolerance search means *none recorded*, not *no allergies*.
- Notes can contradict coded data: a 2021 note says "No Known Allergies" for a patient with 8
  recorded allergies. Safety-critical lists come from the coded resources.
- Notes date quickly: the same note gives the patient's age as of 2021.
