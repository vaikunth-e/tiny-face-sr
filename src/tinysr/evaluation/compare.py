"""Frozen same-scene comparisons and balanced, blinded pairwise review sheets."""

from __future__ import annotations

import csv
import itertools
import json
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from tinysr.common.io import sha256, write_json
from tinysr.models.registry import model_records, selected_artifact
from tinysr.models.runtime import LiteResidual, TorchResidual
from tinysr.evaluation.compose import compose, cost
from tinysr.common.device import environment, select_device
from tinysr.common.geometry import clamp


def resolve(base, path):
    """Resolve explicit manifest paths without guessing sibling file associations."""
    p = Path(path)
    return p if p.is_absolute() else base / p


def load_scenes(path):
    """Require face confirmation and source identity before counting a quality scene."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") not in (
        "tinysr.scenes.v1",
        "tinysr2.scenes.v1",
    ) or not data.get("scenes"):
        raise ValueError("Expected a nonempty TinySR scenes manifest")
    ids, digests = set(), set()
    for row in data["scenes"]:
        scene = row["scene_id"]
        if (
            not scene.isascii()
            or not scene.replace("_", "").replace("-", "").isalnum()
            or scene in ids
        ):
            raise ValueError(
                "Use unique anonymous scene IDs containing letters, numbers, _ or -"
            )
        ids.add(scene)
        if row.get("face_confirmed_by_operator") is not True:
            raise ValueError("Face confirmation is required: " + scene)
        if tuple(row["roi"]) != clamp(row["roi"]):
            raise ValueError("ROI must be inside QVGA: " + scene)
        source = resolve(path.parent, row["source"])
        digest = sha256(source)
        if digest in digests:
            raise ValueError("Duplicate camera input would inflate scene counts")
        digests.add(digest)
        if digest != row["source_sha256_local"]:
            raise ValueError("Frozen camera source changed: " + scene)
        with Image.open(source) as im:
            if im.size != (320, 240):
                raise ValueError("Expected full 320x240 source: " + scene)
        for key in ("camera_bicubic", "saved_c16"):
            if row.get(key):
                control = row[key]
                if control.get("same_source_confirmed_by_operator") is not True:
                    raise ValueError(
                        "Camera control association must be confirmed: " + scene
                    )
                file = resolve(path.parent, control["path"])
                if sha256(file) != control["sha256_local"]:
                    raise ValueError("Frozen camera control changed: " + scene)
                if key == "saved_c16" and control.get("roi") != row["roi"]:
                    raise ValueError("Saved C16 ROI differs from comparison ROI")
                with Image.open(file) as im:
                    if im.size != (1280, 960):
                        raise ValueError("Camera control must be 1280x960")
    return data, path.parent


def comparison_sheet(a, b, roi):
    """Show equal full-frame displays and matching 1:1 face crops without model labels."""
    x, y, w, h = roi
    region = (x * 4, y * 4, (x + w) * 4, (y + h) * 4)
    width = max(640, w * 4)
    sheet = Image.new("RGB", (width * 2, 540 + h * 4), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate((a, b)):
        draw.text((i * width + 12, 10), "A" if i == 0 else "B", fill="black")
        sheet.paste(im.resize((640, 480), Image.Resampling.BOX), (i * width, 40))
        sheet.paste(im.crop(region), (i * width, 540))
    return sheet


def compare(manifest, output, backend="int8", seed=1337, device="auto"):
    """Run offline inference only; every scene keeps its native source scale and own ROI."""
    data, base = load_scenes(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output folder to preserve prior experiments")
    # load verified models before creating a results folder
    models = {}
    for record in model_records():
        name = record["id"]
        if backend == "int8":
            models[name] = LiteResidual(
                selected_artifact(record, "residual_int8.tflite")
            )
        elif backend == "fp32":
            models[name] = TorchResidual(
                selected_artifact(record, "best.pt"),
                record["channels"],
                device=select_device(device),
            )
        else:
            raise ValueError("Use int8 or fp32")
    output.mkdir(parents=True)
    local = output / "local_only"
    local.mkdir()
    # full hashes and paths stay local; never export this manifest or its digest
    write_json(local / "frozen_scenes.json", data)
    write_json(
        local / "randomization.json",
        {
            "seed": seed,
            "method": "python.Random; per-scene pair shuffle and left/right swap",
        },
    )
    rng = random.Random(seed)
    rows, private_key, review = [], [], []
    pair_index = 0
    for row in data["scenes"]:
        sid, roi = row["scene_id"], row["roi"]
        folder = local / sid
        folder.mkdir()
        with Image.open(resolve(base, row["source"])) as im:
            source = im.convert("RGB")
        if row.get("camera_bicubic"):
            with Image.open(resolve(base, row["camera_bicubic"]["path"])) as im:
                baseline = im.convert("RGB")
            baseline_kind = "saved_camera_bicubic_operator_associated"
        else:
            baseline = source.resize((1280, 960), Image.Resampling.BICUBIC)
            baseline_kind = "Pillow_bicubic_desktop_surrogate"
        pictures = {
            "native_bilinear_display": source.resize(
                (1280, 960), Image.Resampling.BILINEAR
            ),
            "bicubic": baseline,
        }
        for record in model_records():
            name = record["id"]
            prediction, stats = compose(source, baseline, roi, models[name].predict)
            pictures[name] = prediction
            rows.append(
                {
                    "scene_id": sid,
                    "model": name,
                    "baseline_kind": baseline_kind,
                    "roi": roi,
                    **stats,
                    **cost(record["channels"], stats["tiles"]),
                }
            )
        if row.get("saved_c16"):
            with Image.open(resolve(base, row["saved_c16"]["path"])) as im:
                pictures["saved_device_C16"] = im.convert("RGB")
            import numpy as np

            difference = np.abs(
                np.asarray(pictures["saved_device_C16"], dtype=np.int16)
                - np.asarray(pictures["C16"], dtype=np.int16)
            )
            rows.append(
                {
                    "scene_id": sid,
                    "comparison": "desktop_C16_vs_saved_device_C16",
                    "changed_channels": int(np.count_nonzero(difference)),
                    "max_rgb_level_difference": int(difference.max()),
                    "scope": "diagnostic only; interpolation, quantized kernel and RGB rounding may differ",
                }
            )
        for name, im in pictures.items():
            im.save(folder / (name + ".png"))
        # compare each neural candidate to bicubic, plus widths and native display
        pairs = [("bicubic", n) for n in models]
        pairs += list(itertools.combinations(models, 2))
        pairs += [("native_bilinear_display", n) for n in models]
        if "saved_device_C16" in pictures:
            pairs += [("bicubic", "saved_device_C16")]
        rng.shuffle(pairs)
        for first, second in pairs:
            a, b = (first, second) if rng.randrange(2) == 0 else (second, first)
            pair_index += 1
            pid = "pair_%04d" % pair_index
            comparison_sheet(pictures[a], pictures[b], roi).save(
                folder / (pid + ".png")
            )
            private_key.append({"pair_id": pid, "scene_id": sid, "A": a, "B": b})
            review.append(
                {
                    "pair_id": pid,
                    "rater_id": "",
                    "preference": "",
                    "objectionable_artifact_A": "",
                    "objectionable_artifact_B": "",
                }
            )
    with (local / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    write_json(local / "blind_key.json", private_key)
    report = {
        "schema": "tinysr.comparison.v1",
        "status": "inference_complete_review_pending",
        "scene_count": len(data["scenes"]),
        "backend": backend,
        "environment": environment(),
        "models": {n: m.metadata for n, m in models.items()},
        "rows": rows,
        "quality_conclusion": None,
        "real_camera_hr_ground_truth": False,
        "hardware_results": False,
        "blinding": "seeded left/right randomization; key stays local",
        "warnings": [
            "desktop inference does not establish firmware parity",
            "changed channels are not quality scores",
            "camera controls depend on explicit operator association",
            "3-pixel halo is specific to the historical shallow model",
        ],
    }
    write_json(output / "comparison.json", report)
    return report
