from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from src.data.dataset_loader import DatasetLoadError, load_dataset
from src.data.dataset_validator import (
    DatasetValidationError,
    trainable_action_ids,
    validate_dataset,
)


def valid_pair():
    events = [
        {
            "event_id": "E1",
            "session_id": "S1",
            "timestamp": "2026-11-01T10:00:00Z",
            "event_type": "user_message",
            "source_type": "user",
            "source_id": "USER_S1",
            "content": "Read the public demo status.",
            "parent_event_id": None,
            "trust_level": "untrusted",
            "metadata": {"scenario_id": "SC1"},
        },
        {
            "event_id": "E2",
            "session_id": "S1",
            "timestamp": "2026-11-01T10:01:00Z",
            "event_type": "assistant_message",
            "source_type": "assistant",
            "source_id": "ASSISTANT_S1",
            "content": "I will read the mock status.",
            "parent_event_id": "E1",
            "trust_level": "trusted",
            "metadata": {"scenario_id": "SC1"},
        },
        {
            "event_id": "E3",
            "session_id": "S1",
            "timestamp": "2026-11-01T10:02:00Z",
            "event_type": "proposed_action",
            "source_type": "assistant",
            "source_id": "ASSISTANT_S1",
            "content": "Read the mock status.",
            "parent_event_id": "E2",
            "trust_level": "untrusted",
            "metadata": {
                "scenario_id": "SC1",
                "requested_by_event_id": "E2",
                "action_type": "mock_read",
                "arguments": {"resource": "status"},
            },
        },
    ]
    manifest = [
        {
            "session_id": "S1",
            "scenario_id": "SC1",
            "attack_type": "none",
            "session_label": "SAFE",
            "session_rationale": "The action matches the user request.",
            "session_review_status": "UNREVIEWED",
            "actions": [
                {
                    "event_id": "E3",
                    "action_label": "SAFE",
                    "action_rationale": "The action is in scope.",
                    "review_status": "UNREVIEWED",
                }
            ],
        }
    ]
    return events, manifest


class DatasetValidatorTests(unittest.TestCase):

    def test_loader_reads_and_validates_separate_jsonl_files(self):
        events, manifest = valid_pair()
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            manifest_path = Path(directory) / "manifest.jsonl"
            events_path.write_text(
                "".join(json.dumps(row) + "\n" for row in events),
                encoding="utf-8",
            )
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            dataset = load_dataset(events_path, manifest_path)
        self.assertEqual(len(dataset.events), 3)
        self.assertEqual(len(dataset.manifest), 1)
        self.assertNotIn("action_label", dataset.events[-1])

    def test_loader_reports_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            manifest_path = Path(directory) / "manifest.jsonl"
            events_path.write_text("{invalid}\n", encoding="utf-8")
            manifest_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(DatasetLoadError, "invalid JSON"):
                load_dataset(events_path, manifest_path)

    def test_valid_pair_keeps_labels_in_manifest(self):
        events, manifest = valid_pair()
        self.assertTrue(validate_dataset(events, manifest))
        self.assertTrue(all("action_label" not in event for event in events))
        self.assertEqual(trainable_action_ids(manifest), set())

    def test_malformed_event_values_raise_validation_error(self):
        cases = (
            ("event_type", [], "event_type is not supported"),
            ("trust_level", {}, "trust_level must be"),
            ("source_type", "", "source_type must be"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                events, manifest = valid_pair()
                events[0][field] = value
                with self.assertRaisesRegex(DatasetValidationError, message):
                    validate_dataset(events, manifest)

    def test_rejects_duplicate_event_ids(self):
        events, manifest = valid_pair()
        events[1]["event_id"] = events[0]["event_id"]
        with self.assertRaisesRegex(DatasetValidationError, "duplicate event_id"):
            validate_dataset(events, manifest)

    def test_rejects_invalid_and_out_of_order_timestamps(self):
        events, manifest = valid_pair()
        events[2]["timestamp"] = "not-a-timestamp"
        with self.assertRaisesRegex(DatasetValidationError, "timestamp must be valid"):
            validate_dataset(events, manifest)

        events, manifest = valid_pair()
        events[1]["timestamp"] = "2026-11-01T09:59:00Z"
        with self.assertRaisesRegex(DatasetValidationError, "chronological order"):
            validate_dataset(events, manifest)

    def test_rejects_missing_and_duplicate_action_references(self):
        events, manifest = valid_pair()
        manifest[0]["actions"][0]["event_id"] = "MISSING"
        with self.assertRaisesRegex(
            DatasetValidationError, "does not reference a proposed_action"
        ):
            validate_dataset(events, manifest)

        events, manifest = valid_pair()
        manifest[0]["actions"].append(deepcopy(manifest[0]["actions"][0]))
        with self.assertRaisesRegex(
            DatasetValidationError, "exactly one manifest label"
        ):
            validate_dataset(events, manifest)

    def test_rejects_unsupported_labels_and_statuses_without_type_errors(self):
        events, manifest = valid_pair()
        manifest[0]["actions"][0]["action_label"] = []
        manifest[0]["actions"][0]["review_status"] = {}
        with self.assertRaisesRegex(DatasetValidationError, "unsupported action_label"):
            validate_dataset(events, manifest)

    def test_review_gate_rejects_unreviewed_binary_labels(self):
        events, manifest = valid_pair()
        with self.assertRaisesRegex(
            DatasetValidationError, "not independently reviewed"
        ):
            validate_dataset(events, manifest, require_reviewed=True)

    def test_ambiguous_action_requires_adjudication_status(self):
        events, manifest = valid_pair()
        manifest[0]["actions"][0].update(
            action_label="AMBIGUOUS", review_status="UNREVIEWED"
        )
        manifest[0]["session_label"] = "AMBIGUOUS"
        manifest[0]["session_review_status"] = "NEEDS_ADJUDICATION"
        with self.assertRaisesRegex(
            DatasetValidationError, "ambiguous label requires adjudication"
        ):
            validate_dataset(events, manifest)


if __name__ == "__main__":
    unittest.main()
