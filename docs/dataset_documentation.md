# Pilot dataset documentation

## Version and scope

Dataset version: 0.2 draft. This is a synthetic format and labeling pilot, not a training-grade dataset. It includes proposed actions in every session and repeats each session scenario ID in event metadata for grouping.

## Current composition

| Measure | Current pilot |
|---|---:|
| Sessions | 10 |
| Distinct scenarios | 10 |
| Events | 39 |
| Proposed actions | 10 |
| SAFE actions | 4 |
| MALICIOUS actions | 4 |
| AMBIGUOUS actions | 2 |
| Independently reviewed actions | 0 |
| Actions awaiting adjudication | 2 |

Event types: 11 user_message, 12 assistant_message, 4 retrieved_context, 10 proposed_action, 1 tool_call, and 1 tool_result. Event trust values: 12 trusted and 27 untrusted. Four sessions contain retrieved context; one contains a tool call.

## Generation guide comparison

The attached generation guide defines an acceptable pilot as at least 300 sessions, 3,000 events, 600 labeled actions, and 30 scenarios. This draft has 10 sessions, 39 events, 10 actions, and 10 scenarios, so it is below every minimum. Treat it as a smoke-test fixture only. Do not use it to claim model performance or generalization.

## Provenance and labels

All examples are synthetic and manually authored. Identifiers are fictional, addresses use example.test, and all actions are simulated. Labels live in the manifest and join to proposed_action events by event_id. Every action and session remains UNREVIEWED except the two AMBIGUOUS cases, which are marked NEEDS_ADJUDICATION. A second teammate must independently review the labels before training or evaluation.

Only independently reviewed SAFE or MALICIOUS actions are eligible for binary training. AMBIGUOUS actions remain available for adjudication and are excluded from binary training.

## Schema and scenario grouping

The Phase 1 graph contract preserves exactly ten canonical top-level event fields. The canonical schema has no top-level scenario_id. To support scenario grouping without changing that contract, each pilot event repeats its manifest scenario_id under metadata.scenario_id; the manifest remains authoritative. Confirm this metadata convention with the ingestion and graph owners before treating it as a finalized shared contract.

## Splits and limitations

No train/validation/test split is published. Future splits must keep each scenario_id in exactly one split to prevent template leakage. Ten scenarios are insufficient for a meaningful evaluation split.

The pilot is small, synthetic, and narrow. It may not reflect production traffic, attack prevalence, linguistic diversity, or tool behavior. Expand the dataset and complete independent review before comparative evaluation.

## Contract dependencies

The event schema and graph contract are described in docs/graph_schema.md and docs/dataset_labeling_guidelines.md. Confirm JSONL ingestion and metadata compatibility with Member 2, and prediction and metric handling with Member 4.
