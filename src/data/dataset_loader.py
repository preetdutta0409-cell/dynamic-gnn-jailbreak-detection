"""Load Canonical Event records and their separate label manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonRecord = dict[str, Any]


class DatasetLoadError(ValueError):
    """Raised when a JSONL dataset file cannot be read."""


@dataclass(frozen=True)
class Dataset:
    """Events and manifest records loaded from JSONL files."""

    events: tuple[JsonRecord, ...]
    manifest: tuple[JsonRecord, ...]


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    """Read non-empty JSONL lines and require each line to contain an object."""
    source = Path(path)
    records: list[JsonRecord] = []

    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetLoadError(
                        f"{source}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise DatasetLoadError(
                        f"{source}:{line_number}: each JSONL record must be an object"
                    )
                records.append(record)
    except OSError as exc:
        raise DatasetLoadError(f"cannot read {source}: {exc}") from exc

    return records


def load_events(path: str | Path) -> list[JsonRecord]:
    """Load event records from a JSONL file."""
    return read_jsonl(path)


def load_manifest(path: str | Path) -> list[JsonRecord]:
    """Load session label records from a JSONL file."""
    return read_jsonl(path)


def load_dataset(
    events_path: str | Path,
    manifest_path: str | Path,
    *,
    require_reviewed: bool = False,
) -> Dataset:
    """Load and structurally validate event and manifest JSONL files.

    Set require_reviewed=True before using binary labels for training. Ambiguous
    actions remain in the returned manifest but are never training examples.
    """
    from src.data.dataset_validator import validate_dataset

    events = load_events(events_path)
    manifest = load_manifest(manifest_path)
    validate_dataset(events, manifest, require_reviewed=require_reviewed)
    return Dataset(events=tuple(events), manifest=tuple(manifest))


def main() -> None:
    """Validate a JSONL event/manifest pair from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and validate dataset JSONL files.")
    parser.add_argument("--events", required=True, help="Path to event JSONL")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Reject SAFE/MALICIOUS actions that lack independent review",
    )
    args = parser.parse_args()
    dataset = load_dataset(args.events, args.manifest, require_reviewed=args.require_reviewed)
    action_count = sum(len(row.get("actions", [])) for row in dataset.manifest)
    print(
        f"Valid dataset: {len(dataset.events)} events, "
        f"{len(dataset.manifest)} sessions, {action_count} labeled actions."
    )


if __name__ == "__main__":
    main()
