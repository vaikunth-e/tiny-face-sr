"""Bind saved camera frames, explicit ROIs, and optional controls into a local manifest."""

import argparse
import csv
from pathlib import Path

from tinysr.common.io import sha256, write_json
from tinysr.evaluation.compare import load_scenes


def freeze(table, output):
    table, output = Path(table).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            "Freeze a new manifest instead of changing a prior experiment"
        )
    scenes = []
    with table.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):

            def path(field):
                p = Path(row[field])
                return p if p.is_absolute() else table.parent / p

            source = path("source")
            roi = [int(row[k]) for k in ("x", "y", "width", "height")]
            scene = {
                "scene_id": row["scene_id"],
                "source": str(source),
                "source_sha256_local": sha256(source),
                "roi": roi,
                "face_confirmed_by_operator": row["face_confirmed"].strip().lower()
                == "yes",
            }
            for key in ("camera_bicubic", "saved_c16"):
                if row.get(key, "").strip():
                    file = path(key)
                    scene[key] = {
                        "path": str(file),
                        "sha256_local": sha256(file),
                        "same_source_confirmed_by_operator": row.get(
                            "controls_confirmed", ""
                        )
                        .strip()
                        .lower()
                        == "yes",
                    }
                    if key == "saved_c16":
                        scene[key]["roi"] = roi
            scenes.append(scene)
    document = {"schema": "tinysr.scenes.v1", "scenes": scenes}
    # validate before publishing the chosen output path
    temporary = output.with_name(output.name + ".validation.json")
    if temporary.exists():
        raise FileExistsError("Temporary validation path already exists")
    try:
        write_json(temporary, document)
        load_scenes(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print("Frozen", len(scenes), "local camera scenes; image hashes remain local")
