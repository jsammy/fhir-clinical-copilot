# Glossary

Terms used across **fhir-clinical-copilot** — its code, design documents, model card and risk
table. Each entry gives what the term means and where it shows up in this system. Entries
describe the system as designed; the README's layer table shows what is implemented so far.

The project runs on synthetic data and is not a medical device. Regulatory terms are included
because the design deliberately borrows their discipline (intended use, risk tables, change
control, traceability), not because the system is subject to them.

---

## 1. Regulatory and quality

| Term | What it is | Where it applies in this system |
|---|---|---|
| **SaMD** (Software as a Medical Device) | Software intended for a medical purpose that is not part of a hardware device | A clinician-facing summariser can cross this line; the intended-use statement is written to make clear this project does not |
| **Intended use / indications for use** | The formal statement of what the software is for, for whom, in what setting | Opens the model card; risk, validation and claims all follow from it |
| **Class I / II / III** | FDA risk classes for devices (low → high) | Framing for the risk table, not a claimed classification |
| **510(k)** | Submission showing substantial equivalence to an existing marketed device | The most common route for AI-enabled devices |
| **De Novo** | Route for a novel low/moderate-risk device with no predicate | How new AI device categories get cleared |
| **PMA** (Premarket Approval) | The most stringent route, for Class III | Listed for completeness |
| **PCCP** (Predetermined Change Control Plan) | A pre-authorised envelope describing how a model may be updated after clearance, and how each update is validated | Modelled by the PCCP-style change note: which changes (prompt, model version, retrieval settings) are allowed and which eval gate each must pass |
| **TPLC** (Total Product Life Cycle) | FDA framing that oversight spans design → deployment → monitoring → retirement | Why monitoring and scheduled evaluation are part of the design, not an afterthought |
| **GMLP** (Good Machine Learning Practice) | FDA / Health Canada / MHRA's ten guiding principles for ML in devices | Used as a checklist when writing the model card |
| **Locked vs adaptive algorithm** | Fixed behaviour vs one that continues learning in the field | This system is locked: model and prompt versions are pinned and changed only through the change note |
| **IEC 62304** | Standard for medical device software lifecycle processes | Informs how components are documented and tested |
| **Safety class A/B/C** | A: no injury possible; B: non-serious injury; C: death or serious injury | Used to reason about which components need the most rigour (generation and citation assembly) |
| **SOUP** (Software of Unknown Provenance) | Third-party code not developed under the manufacturer's lifecycle process | Every open-source library and foundation model here is SOUP; they are inventoried with pinned versions |
| **ISO 13485** | Quality management system standard for medical devices | Context only; this project has no QMS |
| **ISO 14971** | Risk management for medical devices | Source of the hazard → harm → mitigation structure of the risk table |
| **ISO/IEC 42001** | Management system standard for AI | Context for AI governance |
| **Design controls** | Formal traceability from requirements → design → verification → validation | Mirrored by linking each requirement to a test and an eval case |
| **Verification vs validation** | Did we build it right vs did we build the right thing | Unit/integration tests verify; the golden Q&A set validates |
| **Essential performance** | The performance whose loss creates unacceptable risk | Defined here as citation correctness and refusal of ungroundable questions; monitoring alerts on these |
| **Post-market surveillance** | Ongoing collection of real-world performance and complaints | Approximated by scheduled eval runs and drift dashboards |
| **MDR / CE mark** | EU Medical Device Regulation and its conformity mark | The EU route differs from the FDA route |
| **Notified Body** | Third-party organisation that assesses conformity in the EU | The EU counterpart to an FDA reviewer |
| **EU AI Act** | EU regulation classifying AI systems by risk | Medical-device AI generally lands in the high-risk category, on top of MDR |
| **QMS** (Quality Management System) | The documented procedures a manufacturer operates under | Context only |
| **CAPA** | Corrective and Preventive Action | Pattern followed when an eval regression is found: root cause, fix, new golden case |
| **Traceability matrix** | Mapping requirement → design → test → risk | Kept alongside the risk table |

## 2. Privacy and data protection

| Term | What it is | Where it applies in this system |
|---|---|---|
| **PHI** (Protected Health Information) | Health data linked to an identifiable individual, under HIPAA | None is used. The system is still designed as though it were: audit logging, minimum-necessary retrieval, no prompts logged outside the audit store |
| **PII** | Personally identifiable information generally | Broader than PHI |
| **HIPAA** | US law governing PHI use and disclosure | Privacy Rule (use/disclosure), Security Rule (safeguards), Breach Notification Rule |
| **Minimum necessary** | HIPAA principle: use only the least PHI needed | Retrieval sends only the relevant chunks to the model, never the whole record |
| **Safe Harbor de-identification** | Removing 18 specified identifier types | The mechanical de-identification route |
| **Expert Determination** | A qualified expert certifies re-identification risk is very small | The statistical route |
| **Limited Data Set / DUA** | Partially de-identified data released under a Data Use Agreement | The access model for credentialed datasets such as MIMIC-IV |
| **BAA** (Business Associate Agreement) | Contract binding a vendor that handles PHI | Required before any cloud AI service could touch real PHI |
| **GDPR special category data** | Health data under EU law, requiring a stronger legal basis | Relevant to any EU deployment |
| **DPDP Act 2023** | India's Digital Personal Data Protection Act | Relevant to any India deployment |
| **Data residency** | Requirement that data remain in a jurisdiction | Drives region choice and whether cross-region inference is allowed |
| **Re-identification** | Linking de-identified data back to a person | Prohibited by every DUA |

## 3. FHIR and interoperability

| Term | What it is | Where it applies in this system |
|---|---|---|
| **HL7** | The standards body (and the older v2 messaging standard) | HL7 v2 messages still carry most in-hospital traffic |
| **FHIR** | Fast Healthcare Interoperability Resources — REST + JSON standard for health data | The system's data model and source of truth |
| **R4 / STU3 / DSTU2 / R5** | FHIR versions | R4 is used throughout |
| **Resource** | A single typed object (Patient, Observation…) with an ID | Retrieval chunks keep resource IDs so every answer can cite them |
| **Bundle** | A collection of resources; the `transaction` type is applied atomically | Synthea emits transaction bundles, POSTed whole to the server root |
| **Patient / Encounter / Condition / Observation / MedicationRequest / DiagnosticReport** | The resources covering most clinical questions | The resource types the retrieval layer indexes first |
| **Reference** | A pointer from one resource to another (`Patient/123`) | Used to rebuild a patient timeline during flattening |
| **Profile** | A constrained version of a resource for a use case | Real servers validate against profiles |
| **US Core** | The US profile set built on R4 | Synthea output follows it |
| **Extension** | Standard mechanism for adding non-standard fields | Preserved during flattening rather than dropped |
| **Search parameter** | Query key: `Observation?patient=X&code=Y&date=ge2025-01-01` | How the agent's read-only tools query the FHIR server |
| **Chained search / `_include`** | Traversing references within one query | Avoids N+1 request patterns in the agent tools |
| **SMART on FHIR** | OAuth2 profile for FHIR app authorisation | The production route for scoped clinician access; out of scope for the PoC |
| **Scopes** | `patient/Observation.read`-style permissions | The agent is limited to read scopes |
| **Bulk Data / Flat FHIR (`$export`)** | NDJSON export of large populations | The route for loading a real cohort at scale |
| **CDS Hooks** | Standard for injecting decision support into EHR workflow | The natural production integration point for a copilot |
| **C-CDA** | XML clinical document format | The older document-based exchange route |
| **IHE** | Profiles that combine standards into workflows (XDS, PIX) | Common in imaging and hospital integration |

## 4. Clinical vocabularies

| Term | What it is | Where it applies in this system |
|---|---|---|
| **LOINC** | Codes for lab tests and measurements | Identifies what was measured in an Observation |
| **SNOMED CT** | Clinical terminology for conditions, findings, procedures | Identifies diagnoses in Condition resources |
| **ICD-10 / ICD-10-CM** | Diagnosis classification, billing-oriented | What claims data uses |
| **CPT / HCPCS** | Procedure and service codes | Billing side of procedures |
| **RxNorm** | Normalised drug names | Codes in MedicationRequest resources |
| **NDC** | National Drug Code, product-level | More granular than RxNorm |
| **UCUM** | Unit codes (mg/dL, mmHg) | Units are carried into chunks and answers; dropped units are a silent error source |
| **Terminology server / ValueSet / CodeSystem** | Services and artefacts for resolving codes | Translating a code into text a model can read |
| **Code system + code + display** | The triple FHIR uses for any coded value | Answers cite the code, not only the display text |

## 5. Imaging

| Term | What it is | Where it applies in this system |
|---|---|---|
| **DICOM** | Standard for medical images plus embedded metadata | Outside the current scope; noted for a later de-identification exercise |
| **PACS** | Picture Archiving and Communication System | Where images live in a hospital |
| **Modality** | The acquisition type: CT, MR, US, CR, DX, MG | A DICOM tag |
| **Study / Series / Instance** | DICOM hierarchy: visit → acquisition run → single image | The model for any imaging query |
| **SOP Class / SOP Instance UID** | Object type and unique identifier | Primary keys of the imaging world |
| **DICOMweb (WADO-RS, QIDO-RS, STOW-RS)** | REST APIs for retrieve, query and store | The cloud-friendly access route |
| **Hanging protocol** | Rules for how images are laid out for reading | Radiologist workflow vocabulary |
| **Windowing / window level** | Mapping raw or Hounsfield values to display grey levels | Why raw pixels are not what the radiologist sees |
| **De-identification (DICOM)** | Stripping identifiers from tags, and burned-in text from pixels | Pixel-level PHI is the commonly missed case |
| **MONAI** | PyTorch framework for medical imaging | Reference framework for imaging work |
| **Radiology report** | The free-text narrative accompanying a study | Where NLP meets imaging |

## 6. GenAI, agents, evaluation

| Term | What it is | Where it applies in this system |
|---|---|---|
| **RAG** | Retrieval-augmented generation: fetch context, then generate | The core pattern of the Q&A path |
| **Grounding / groundedness** | Whether every claim traces to retrieved source text | The primary evaluation metric |
| **Faithfulness vs relevance** | Answer true to sources vs answer useful to the question | Both are scored; they fail differently |
| **Hallucination** | Fluent output unsupported by any source | Treated as a safety hazard in the risk table, not a quality bug |
| **Citation / attribution** | Linking each claim to a resource ID | Required on every answer; uncitable claims are refused |
| **Chunking** | Splitting documents for embedding | Chunked by patient and clinical episode, not by character count |
| **Embedding / vector store** | Dense representation and its index | Bedrock embeddings into S3 Vectors (or OpenSearch) |
| **Hybrid search / reranking** | Combining keyword and semantic retrieval, then reordering | Clinical codes are exact-match; pure semantic search misses them |
| **Context window** | Token budget for a single request | Bounds how much patient history reaches the model |
| **Tool / function calling** | Model invoking a defined function with structured arguments | How the agent reaches FHIR |
| **MCP** (Model Context Protocol) | Open standard for exposing tools and context to models | Read-only FHIR queries are exposed as MCP tools |
| **Agent / agentic** | System that plans and acts across multiple steps | The pre-visit summary workflow |
| **Human in the loop** | Required human approval before an output takes effect | The pre-visit summary is a draft until a clinician approves it |
| **Guardrails** | Input/output filters, allow-lists, refusal policies | Bedrock Guardrails plus application-level refusal rules |
| **Prompt injection** | Malicious instructions hidden in retrieved content | Retrieved text is treated as data, never as instructions; covered by adversarial eval cases |
| **Golden set / eval suite** | Fixed Q&A pairs with expected behaviour | The release gate: a change that regresses it does not ship |
| **LLM-as-judge** | Using a model to score outputs | Used for groundedness scoring, spot-checked against manual review |
| **Drift** | Degradation as data or model changes | Tracked by scheduled eval runs over time |
| **Model card** | Disclosure document: intended use, data, metrics, limitations | Published in `docs/` |
| **Fine-tuning vs prompting vs RAG** | Changing weights vs instructions vs supplied context | This system uses prompting + RAG only; no weights are changed |
| **Distillation / quantisation** | Shrinking models for cost or latency | Relevant to edge inference; not used here |

## 7. Classical ML and deep learning

| Term | What it is | Healthcare use |
|---|---|---|
| **CNN** | Convolutional network exploiting spatial locality | Imaging classification and segmentation |
| **U-Net** | Encoder-decoder CNN with skip connections | The standard for medical image segmentation |
| **RNN / LSTM / GRU** | Sequence models with memory | ICU vitals, time-series deterioration prediction |
| **Transformer / attention** | Sequence model using self-attention | Clinical notes, and increasingly imaging (ViT) |
| **VAE** | Variational autoencoder: learns a latent distribution | Synthetic data generation, anomaly detection |
| **GAN** | Two networks in competition | Image synthesis, modality translation, augmentation |
| **Diffusion model** | Iterative denoising generation | Current state of the art for image synthesis |
| **Sensitivity / specificity** | True positive rate / true negative rate | The clinical framing of recall; preferred over accuracy |
| **PPV / NPV** | Predictive values, prevalence-dependent | Why a model that works in trials can fail in a low-prevalence clinic |
| **AUROC / AUPRC** | Ranking quality; AUPRC is better for rare outcomes | Most clinical outcomes are rare |
| **Calibration** | Whether predicted probabilities match observed rates | Often more important clinically than discrimination |
| **Class imbalance** | Rare positives dominating error behaviour | The default condition in clinical data |
| **Subgroup / fairness analysis** | Performance broken down by age, sex, site, ethnicity | Eval results are reported by age band and sex |
| **Distribution shift** | Deployment data differing from training data | Different scanner, hospital, or year |
| **Explainability (SHAP, Grad-CAM)** | Attribution methods for tabular and image models | Citations play this role for generated answers |

## 8. Platform, cloud and MLOps

| Term | What it is | Where it applies in this system |
|---|---|---|
| **MLOps / LLMOps** | Lifecycle practices for ML and LLM systems | Versioned prompts, pinned models, eval-gated changes |
| **Feature store** | Managed, reusable feature storage | Not used; noted for contrast |
| **Model registry** | Versioned model catalogue with stage transitions | Model and prompt versions are recorded with each eval run |
| **Shadow deployment / canary** | Running a new model alongside, or on a slice of, traffic | The intended rollout path for a model change |
| **Amazon Bedrock** | Managed foundation models on AWS | The generation and embedding layer |
| **Bedrock AgentCore** | AWS managed agent runtime | Alternative host for the agent layer |
| **SageMaker** | AWS training, pipelines and hosted inference | Not needed for the current scope |
| **S3 Vectors** | Vector storage in S3 | Chosen for cost; lower throughput than OpenSearch |
| **AWS HealthLake** | Managed FHIR data store on AWS | The managed alternative to running HAPI FHIR |
| **Aurora Serverless** | Managed relational DB that scales to demand | Holds the audit log; accessed through RDS Proxy to avoid Lambda connection exhaustion |
| **Lambda / SQS / EventBridge / Step Functions** | Serverless compute, queue, event bus, orchestration | Ingestion pipeline and scheduled evals; DLQs and visibility timeouts configured explicitly |
| **Multi-tenancy / tenant isolation** | Serving multiple customers from one platform safely | Out of scope for the PoC; the audit log is keyed so it could be partitioned by tenant |
| **OpenTelemetry** | Vendor-neutral telemetry standard | Traces latency, token cost and eval scores per request |
| **SLO / error budget** | Reliability target and the allowance for missing it | Defined for answer latency and groundedness |
| **IaC (Terraform / CDK)** | Infrastructure as code | Target infrastructure is defined as code |
| **Least privilege** | Grant only the permissions needed | Each Lambda role is scoped to the resources it touches |
| **Provenance / SBOM** | Build attestation and dependency inventory | Doubles as the SOUP inventory |

## 9. Datasets and clinical context

| Term | What it is | Where it applies in this system |
|---|---|---|
| **Synthea** | Synthetic patient generator producing FHIR R4 | The primary data source; no privacy constraints |
| **MIMIC-IV** | De-identified ICU and ED data from Beth Israel Deaconess | Reference for realistic data messiness; the demo subset is open access |
| **PhysioNet** | The repository hosting MIMIC and related datasets | Credentialing gate for the full datasets |
| **eICU** | Multi-centre ICU database | Reference for cross-site generalisation |
| **EHR / EMR** | Electronic health/medical record system | FHIR is the standard route into them |
| **Encounter** | A single clinical contact | The unit most summarisation questions hinge on |
| **Problem list / medication reconciliation** | Active diagnoses; reconciling meds across care transitions | Two of the pre-visit summary's sections |
| **Discharge summary** | Narrative handover at end of stay | The canonical clinical summarisation target |
| **Clinical workflow integration** | Fitting into how clinicians actually work | Why the summary is a draft for approval, not an automatic output |
