"""Review completeness, scene weighting and export privacy regressions"""

import csv, json, zipfile
from pathlib import Path
import pytest
from tinysr.common.io import write_json
from tinysr.evaluation.review import summarize_review
from tinysr.evaluation.export import export_metrics
from tinysr.evaluation.compare import compare, load_scenes
from tinysr.data.scenes import freeze
from PIL import Image


def review_case(root):
    local = root / "local_only"
    local.mkdir(parents=True)
    write_json(root / "comparison.json", {"scene_count": 2})
    write_json(
        local / "blind_key.json",
        [
            {"pair_id": "p1", "scene_id": "s1", "A": "C16", "B": "bicubic"},
            {"pair_id": "p2", "scene_id": "s2", "A": "bicubic", "B": "C16"},
            {"pair_id": "p3", "scene_id": "s2", "A": "C24", "B": "C16"},
        ],
    )
    return local


def test_wins_ties_losses_and_direct_pairs(tmp_path):
    local = review_case(tmp_path)
    (local / "review.csv").write_text(
        "pair_id,rater_id,preference,objectionable_artifact_A,objectionable_artifact_B\np1,r1,A,no,no\np2,r1,tie,no,no\np3,r1,A,no,yes\n"
    )
    result = summarize_review(tmp_path)
    assert result["status"] == "complete"
    pairs = {(p["first"], p["second"]): p for p in result["pairs"]}
    assert pairs["C16", "bicubic"]["first_wins"] == 1
    assert pairs["C16", "bicubic"]["ties"] == 1
    assert pairs["C16", "bicubic"]["scene_mean_first_win_rate"] == 0.5
    assert pairs["C16", "C24"]["second_wins"] == 1
    assert pairs["C16", "C24"]["artifacts"]["C16"] == 1


def test_missing_rows_are_pending(tmp_path):
    local = review_case(tmp_path)
    (local / "review.csv").write_text(
        "pair_id,rater_id,preference,objectionable_artifact_A,objectionable_artifact_B\np1,r1,A,no,no\n"
    )
    result = summarize_review(tmp_path)
    assert result["status"] == "partial" and result["pending_rows"] == 2


def test_exports_exclude_injected_private_content(tmp_path):
    write_json(
        tmp_path / "review_summary.json",
        {
            "scene_count": 1,
            "status": "partial",
            "pending_rows": 1,
            "environment": {"hostname": "PRIVATE_MARKER"},
            "source_sha256_local": "PRIVATE_MARKER",
            "utc": "PRIVATE_MARKER",
            "path": "PRIVATE_MARKER",
            "pairs": [],
        },
    )
    path = export_metrics(tmp_path, tmp_path / "metrics.zip")
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["metrics.json"]
        assert b"PRIVATE_MARKER" not in z.read("metrics.json")


def test_actual_int8_workflow_and_freeze(tmp_path):
    Image.new("RGB", (320, 240), (100, 110, 120)).save(tmp_path / "source.png")
    fields = [
        "scene_id",
        "source",
        "x",
        "y",
        "width",
        "height",
        "face_confirmed",
        "controls_confirmed",
    ]
    with (tmp_path / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            dict(
                zip(fields, ["scene_001", "source.png", 110, 50, 140, 140, "yes", "no"])
            )
        )
    manifest = tmp_path / "scenes.json"
    freeze(tmp_path / "table.csv", manifest)
    report = compare(manifest, tmp_path / "run")
    assert all(row["tiles"] == 9 for row in report["rows"])
    assert all(row["outside_roi_unchanged"] for row in report["rows"])
    assert summarize_review(tmp_path / "run")["pending_rows"] == 9
    (tmp_path / "source.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_scenes(manifest)
