# Synthetic pilot dataset

Version: 0.3 draft

The current synthetic dataset contains 500 sessions, 5,530 canonical events, 50 scenario families, and 1,000 labeled proposed actions. It meets the generation guide's recommended lower bounds for size. It is not training-ready until independent label review is complete.

## Files

- raw/pilot_events.jsonl: canonical event records, one per line.
- raw/pilot_manifest.jsonl: session and action labels, one manifest record per session.
- splits/scenario_splits.json: deterministic split assignments grouped by scenario_id.
- ../docs/dataset_statistics.md: counts, distributions, review state, generation method, and split sizes.

## Format and review

Events retain exactly ten canonical top-level fields. Each event repeats its scenario ID under metadata.scenario_id; the manifest remains authoritative. Labels stay in the manifest and join to proposed_action events by event_id.

Counts: 524 SAFE, 456 MALICIOUS, and 20 AMBIGUOUS actions. SAFE/MALICIOUS labels are UNREVIEWED. AMBIGUOUS labels require adjudication. Only independently reviewed SAFE or MALICIOUS actions are eligible for binary training.

## Utility commands

Run from the repository root:

- Validate: python -m src.data.dataset_loader --events data/raw/pilot_events.jsonl --manifest data/raw/pilot_manifest.jsonl
- Require reviewed binary labels: add --require-reviewed. It should reject the current draft until independent review is recorded.
- Recreate scenario-grouped split assignments: python -m src.data.split_dataset --manifest data/raw/pilot_manifest.jsonl --output data/splits/scenario_splits.json --seed 42

All examples are synthetic. Tools and side effects are simulated, and the data contains no real secrets or private user data. See docs/dataset_documentation.md and docs/dataset_statistics.md for details.
