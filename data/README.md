# Pilot dataset

Version: 0.1 draft

This directory contains a small, synthetic pilot for Member 1's dataset and evaluation work. It is a format and labeling pilot, not a final training or benchmark dataset.

## Files

- raw/pilot_events.jsonl: one event record per line.
- raw/pilot_manifest.jsonl: one manifest record per session; action labels refer to the event_id of a proposed_action event.

## Format and labeling

Events follow the ten top-level fields in labeling guide v0.3: event_id, session_id, timestamp, event_type, source_type, source_id, content, parent_event_id, trust_level, and metadata. Event IDs are globally unique. Timestamps use UTC ISO 8601 with Z. Labels are stored only in the manifest. Member 2 still needs to confirm ingestion adoption and field compatibility.

The primary target is action_label on proposed_action events. SAFE maps to 0 and MALICIOUS maps to 1 for binary metrics; AMBIGUOUS is excluded from primary binary metrics and requires adjudication. Sessions without proposed actions have no action labels.

Edge-related fields are included where applicable: assistant messages that use retrieved context include metadata.retrieved_context_event_ids; proposed actions include metadata.requested_by_event_id naming the requesting assistant message. No proposed_action-to-tool edge is represented.

## Provenance and permitted use

All examples are manually authored synthetic scenarios. They use fictional identifiers, example.test addresses, and simulated tools only. No external dataset, real personal data, credentials, real email, or real database is included. The pilot is original project material; no external dataset license applies.

## Review and limitations

All non-ambiguous labels are UNREVIEWED until a teammate independently reviews them. The two ambiguous cases are marked NEEDS_ADJUDICATION. Do not use them in primary binary metrics unless they are resolved and relabeled.

The sample is intentionally small and synthetic. It cannot support performance claims or establish generalization. It has no train/validation/test split yet. When the dataset grows, split by scenario_id or another agreed template group, never by event row, and report session/action counts and class balance.

The JSONL layout and metadata details should be checked with Member 2 (ingestion) and Member 4 (prediction/evaluation) before the pilot becomes a larger dataset.
