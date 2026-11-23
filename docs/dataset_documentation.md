# Dataset documentation

## Version and scope

Dataset version: 0.3 draft. This synthetic dataset is prepared for pilot training and evaluation after independent label review. It includes 500 sessions, 5,530 events, 50 scenario families, and 1,000 proposed-action labels.

## Composition

See [dataset_statistics.md](dataset_statistics.md) for event and trust distributions, attack-category counts, review state, and scenario-grouped split sizes.

| Action label | Count |
|---|---:|
| SAFE | 524 |
| MALICIOUS | 456 |
| AMBIGUOUS | 20 |
| Total | 1,000 |

The binary draft class balance is 53.5% SAFE and 46.5% MALICIOUS. All 980 binary labels remain UNREVIEWED; the 20 AMBIGUOUS labels require adjudication. Do not train or report model performance until review is complete.

## Action counts by split

| Split | SAFE | MALICIOUS | AMBIGUOUS | Total |
|---|---:|---:|---:|---:|
| Train | 366 | 324 | 10 | 700 |
| Validation | 94 | 66 | 0 | 160 |
| Test | 64 | 66 | 10 | 140 |

These are draft label counts; no SAFE or MALICIOUS label has independent review.

## Generation and provenance

The data is synthetic and generated from 50 authored scenario families with ten session variations per family. Scenarios cover benign tasks, direct and multi-turn jailbreaks, instruction hierarchy manipulation, prompt extraction, role-play, indirect document, web, and tool-output injection, unauthorized access and tool use, exfiltration, privilege escalation, and ambiguous scope. Tool actions are simulated, addresses use example.test, and no real private data or external side effects are included.

Labels are stored only in the separate manifest and join to proposed_action events using event_id. They were assigned from the project rubric but have not received independent teammate review.

## Schema and graph structure

Events retain the ten canonical top-level fields from the Phase 1 graph contract. scenario_id is repeated under metadata.scenario_id to support split grouping and must match the authoritative scenario_id in the manifest. Confirm this metadata convention with the ingestion and graph owners before treating it as a finalized shared contract.

The dataset includes linked user messages, assistant messages, retrieved context, tool calls/results, proposed actions, and pending policy-decision events. Parent and retrieval references stay within their session. All proposed actions have exactly one manifest label.

## Splits and limitations

Scenario assignments are in data/splits/scenario_splits.json. The split contains 35 train scenarios, 8 validation scenarios, and 7 test scenarios; sessions from one scenario family never cross splits.

The dataset meets the guide's recommended lower bounds for size, but remains fully synthetic and unreviewed. Complete independent label review and adjudication before model training or evaluation.
