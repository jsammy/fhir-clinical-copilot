# AWS failure modes

How the target AWS deployment fails, and what the design does about each failure. Entries are
marked **observed** (hit while building this project) or **anticipated** (not yet seen here;
taken from service quotas and documented behaviour).

## Bedrock

| Failure | Status | Handling |
|---|---|---|
| `ThrottlingException`: tokens-per-minute or requests-per-minute quota exceeded | anticipated | botocore `retries={"mode": "adaptive", "max_attempts": 6}` in `src/agent/llm.py`: client-side rate limiting plus exponential backoff with jitter. The global inference profile spreads load across regions. If retries run out, the narrative is marked unavailable and the code-written sections still go to review. |
| New account blocked during verification (`AccessDeniedException`) | **observed** | Cleared on its own within the stated 2 hours. |
| Model breaks the citation format | **observed** | Nova Micro wrote `[101, 202]`; Haiku wrote two refs in one bracket and two sentences per list item. The validator parses every bracket and checks each sentence. One retry with the problems as feedback, then the narrative is withheld. |
| Model repeats a stale fact from an old note | **observed** | Haiku stated an age copied from a 2021 note. The prompt now forbids demographics and requires notes to be dated when used. A citation check cannot catch this; groundedness scoring in the evaluation can. |
| Region-bound credentials | **observed** | Short-term Bedrock API keys work only in the region that issued them. `AWS_REGION` must match. |
| Model or profile not offered in a region | anticipated | Model id is configuration (`BEDROCK_MODEL_ID`), never code. |

## Lambda

| Failure | Status | Handling |
|---|---|---|
| 15-minute hard timeout | anticipated | Ingest processes one bundle per invocation. Tool calls are page-capped (5 pages), so one request cannot crawl a 4,000-observation record. |
| Concurrency limit (account default 1,000; new accounts can be lower) | anticipated | Reserved concurrency on ingest so a bulk load cannot starve the query API. SQS absorbs bursts. |
| Cold starts on the API | anticipated | Acceptable for a demo. Provisioned concurrency costs money while idle, so it stays off. |
| Lambda in a VPC needs a NAT Gateway to reach Bedrock | anticipated | Lambdas stay outside the VPC. Aurora is reached through the RDS Data API over HTTPS. |

## SQS

| Failure | Status | Handling |
|---|---|---|
| Message reappears while still being processed | anticipated | Visibility timeout set to at least 6× the Lambda timeout, per AWS guidance. |
| Poison message (a bundle that always fails) | anticipated | `maxReceiveCount` 3, then a dead-letter queue. Redrive after the fix. 4xx from the FHIR server is not retried; 5xx is. |
| Duplicate delivery (standard queues are at-least-once) | anticipated | Loader checks the patient identifier before POST (`--skip-existing` logic), since a FHIR transaction POST is not idempotent. |

## Vector store

| Failure | Status | Handling |
|---|---|---|
| S3 Vectors query throughput and latency lower than OpenSearch | anticipated | Fine at demo volume. The swap point is the retrieval module's interface, not the agent. |
| Cross-patient leakage through retrieval | anticipated | Every query carries a `patient_ref` metadata filter taken from the request path, never from the model or the question. |

## Aurora Serverless v2 (audit log)

| Failure | Status | Handling |
|---|---|---|
| Connection exhaustion: many Lambdas each opening a connection | anticipated | RDS Data API: HTTP calls, no connection pool to exhaust, no RDS Proxy cost. |
| Resume from 0 ACU takes ~15 s, longer after a day paused | anticipated | The audit write's timeout allows for the resume. A summary is not reported as released until its audit record is written. |

## Local development

| Failure | Status | Handling |
|---|---|---|
| HAPI lost all data on container restart | **observed** | Named volume for H2. A one-shot busybox container chowns it, because the distroless HAPI image runs as uid 65532. |
| Long-running load killed partway | **observed** | `load_bundles --skip-existing` resumes without duplicating patients. |
