# Pilot dataset documentation

## Version and scope

Dataset version: 0.1 draft. This document describes the synthetic pilot in data/raw/pilot_events.jsonl and data/raw/pilot_manifest.jsonl.

## Composition

| Session label | Sessions | Proposed actions |
|---|---:|---:|
| SAFE | 4 | 1 |
| MALICIOUS | 4 | 4 |
| AMBIGUOUS | 2 | 2 |
| Total | 10 | 7 |

Safe cases cover ordinary question answering, educational discussion, quoted attack text, and a legitimate mock email. Malicious cases cover indirect prompt injection, multi-turn manipulation, tool-result injection, and forged policy authority. Ambiguous cases cover overlapping instructions and unclear query scope.

Primary action metrics currently have five non-ambiguous proposed-action examples: one SAFE and four MALICIOUS. This is far too small for model evaluation; the counts are only a pilot-format check.

## Provenance and generation

All content is synthetic and manually authored for this project. No external dataset or source text is used. Identifiers are fictional, addresses use example.test, and all actions are simulated. There are no real external side effects.

## Labeling and review

Labels follow docs/dataset_labeling_guidelines.md v0.3 on the Member 1 pilot branch. Session and action labels are kept separately in the manifest. Action labels join to proposed_action events by event_id. SAFE and MALICIOUS are the primary binary classes; AMBIGUOUS is excluded until adjudicated.

All labels remain UNREVIEWED except the two ambiguous cases, which are marked NEEDS_ADJUDICATION. A second teammate should independently review the pilot before it is used for training or evaluation.

## Splits and limitations

No split files are included. Keep any future variants of the same scenario template in one split using scenario_id. Do not make random event-row splits.

The pilot is small, synthetic, and intentionally narrow. It may not reflect real production traffic, attack prevalence, linguistic diversity, or tool behavior. Do not report model quality or claim generalization from this pilot. Expand and review the dataset before comparative evaluation.

## Contract dependencies

Confirm the JSONL ingestion representation and event field names with Member 2, and confirm prediction/metric handling with Member 4. Update this documentation and version if the contract changes.
