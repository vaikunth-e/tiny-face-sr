"""Explicit pair outcomes with pending review and scene-weighted aggregation"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from tinysr.common.io import write_json


def summarize_review(folder):
    """Retain wins, ties and losses for every model pair without inflating scene counts"""
    folder = Path(folder)
    report = json.loads((folder / "comparison.json").read_text())
    entries = json.loads((folder / "local_only/blind_key.json").read_text())
    keys = {r["pair_id"]: r for r in entries}
    if len(keys) != len(entries):
        raise ValueError("Duplicate blind pair IDs")
    with (folder / "local_only/review.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    covered = set()
    seen = set()
    pending = 0
    pairs = {}
    for row in rows:
        pid = row["pair_id"]
        if pid not in keys:
            raise ValueError("Unknown blind pair")
        covered.add(pid)
        identity = (pid, row["rater_id"])
        if identity in seen:
            raise ValueError("Duplicate pair and rater")
        seen.add(identity)
        if (
            not row["rater_id"]
            or row["preference"] not in ("A", "B", "tie")
            or any(
                row.get(k) not in ("yes", "no")
                for k in ("objectionable_artifact_A", "objectionable_artifact_B")
            )
        ):
            pending += 1
            continue
        key = keys[pid]
        names = sorted([key["A"], key["B"]])
        first, second = names
        pair = pairs.setdefault(
            first + "__" + second,
            {
                "first": first,
                "second": second,
                "first_wins": 0,
                "ties": 0,
                "second_wins": 0,
                "ratings": 0,
                "artifacts": {first: 0, second: 0},
                "scenes": {},
            },
        )
        winner = None if row["preference"] == "tie" else key[row["preference"]]
        pair[
            (
                "ties"
                if winner is None
                else ("first_wins" if winner == first else "second_wins")
            )
        ] += 1
        pair["ratings"] += 1
        votes = pair["scenes"].setdefault(key["scene_id"], [])
        votes.append(0 if winner is None else (1 if winner == first else -1))
        for side in ("A", "B"):
            pair["artifacts"][key[side]] += (
                row["objectionable_artifact_" + side] == "yes"
            )
    pending += len(set(keys) - covered)
    for pair in pairs.values():
        scene_votes = pair.pop("scenes")
        pair["reviewed_scenes"] = len(scene_votes)
        pair["scene_mean_first_win_rate"] = sum(
            sum(x == 1 for x in v) / len(v) for v in scene_votes.values()
        ) / len(scene_votes)
        pair["scene_mean_tie_rate"] = sum(
            sum(x == 0 for x in v) / len(v) for v in scene_votes.values()
        ) / len(scene_votes)
        pair["scene_mean_second_win_rate"] = sum(
            sum(x == -1 for x in v) / len(v) for v in scene_votes.values()
        ) / len(scene_votes)
    result = {
        "schema": "tinysr.review.v1",
        "status": "partial" if pending else "complete",
        "pending_rows": pending,
        "scene_count": report["scene_count"],
        "pairs": list(pairs.values()),
        "quality_gate_automatically_established": False,
    }
    write_json(folder / "review_summary.json", result)
    return result
