# Pilot dataset

Version: 0.2 draft

This is a small, synthetic smoke-test fixture for Member 1's dataset and evaluation work. It is not a training or benchmark dataset.

## Files

- raw/pilot_events.jsonl: one canonical event record per line.
- raw/pilot_manifest.jsonl: one label record per session; action labels refer to proposed_action event IDs.
- ../docs/dataset_documentation.md: scope, counts, review state, limitations, and generation-guide comparison.

## Format and labeling

Events retain the ten canonical top-level fields. The scenario ID is repeated as metadata.scenario_id for grouped split preparation and must match the authoritative scenario_id in the session manifest. This keeps the Phase 1 top-level schema unchanged; confirm the metadata convention with the ingestion and graph owners.

The primary target is action_label on proposed_action events. SAFE maps to 0 and MALICIOUS maps to 1 only after independent review. AMBIGUOUS is excluded from binary training and metrics until adjudicated. Labels remain in the manifest, never in graph events.

## Provenance and permitted use

All examples are synthetic, use fictional identifiers and example.test addresses, and describe mock actions only. No real email, database, personal data, credentials, or external dataset is included.

## Review and limitations

The pilot contains 10 sessions, 39 events, 10 proposed actions, and 10 scenarios. The generation guide's acceptable-pilot minimums are 300 sessions, 3,000 events, 600 actions, and 30 scenarios. The current pilot is below each threshold. No labels are independently reviewed and no evaluation split is published.

Keep every scenario group within one split. Expand the dataset, obtain independent label review, and publish a quality report before training or making performance claims.

## Utility commands

From the repository root:

- Validate: python -m src.data.dataset_loader --events data/raw/pilot_events.jsonl --manifest data/raw/pilot_manifest.jsonl
- Require independent binary-label review: add --require-reviewed; this draft should be rejected because no binary action labels have independent review.
- Create future scenario-grouped splits: python -m src.data.split_dataset --manifest data/raw/pilot_manifest.jsonl --output data/splits/split_manifest.json --seed 42

Do not create or use evaluation splits until the dataset has enough reviewed scenarios; the split command is provided for a sufficiently large future dataset.
