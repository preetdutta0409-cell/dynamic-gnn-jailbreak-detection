"""Create deterministic train/validation/test assignments by scenario family."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_NAMES = ("train", "validation", "test")


def split_by_scenario(
    manifest: Sequence[Mapping[str, Any]],
    *,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Split whole scenario groups so related sessions cannot cross splits.

    Ratios apply to the number of distinct scenarios. Session counts may differ
    from the requested percentages when scenario families contain unequal
    numbers of sessions.
    """
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0")

    grouped: dict[str, list[str]] = {}
    seen_sessions: set[str] = set()
    for entry in manifest:
        session_id = entry.get("session_id")
        scenario_id = entry.get("scenario_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("each manifest entry needs a non-empty session_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("each manifest entry needs a non-empty scenario_id")
        if session_id in seen_sessions:
            raise ValueError(f"duplicate session_id in manifest: {session_id}")
        seen_sessions.add(session_id)
        grouped.setdefault(scenario_id, []).append(session_id)

    scenario_ids = sorted(grouped)
    if len(scenario_ids) < 3:
        raise ValueError("at least three distinct scenarios are needed for three splits")

    exact = [len(scenario_ids) * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = len(scenario_ids) - sum(counts)
    for index in sorted(
        range(3), key=lambda item: exact[item] - counts[item], reverse=True
    )[:remaining]:
        counts[index] += 1

    for index, count in enumerate(counts):
        if count == 0:
            donor = max(
                (other for other in range(3) if counts[other] > 1),
                key=lambda other: counts[other] - exact[other],
            )
            counts[donor] -= 1
            counts[index] = 1

    random.Random(seed).shuffle(scenario_ids)
    output: dict[str, dict[str, list[str]]] = {}
    start = 0
    for name, count in zip(SPLIT_NAMES, counts):
        assigned_scenarios = scenario_ids[start : start + count]
        start += count
        session_ids = sorted(
            session_id
            for scenario_id in assigned_scenarios
            for session_id in grouped[scenario_id]
        )
        output[name] = {
            "scenario_ids": sorted(assigned_scenarios),
            "session_ids": session_ids,
        }

    return output


def write_split_manifest(
    splits: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    *,
    seed: int,
    ratios: Sequence[float],
) -> None:
    """Write one JSON split manifest; no event rows are duplicated."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "splits": splits,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Create a JSON split manifest grouped by scenario_id."""
    import argparse

    from src.data.dataset_loader import load_manifest

    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test scenario splits."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--output", required=True, help="Output JSON split manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    splits = split_by_scenario(manifest, seed=args.seed)
    write_split_manifest(splits, args.output, seed=args.seed, ratios=(0.70, 0.15, 0.15))
    print(f"Wrote scenario-grouped splits for {len(manifest)} sessions to {args.output}")


if __name__ == "__main__":
    main()
