"""Score reviewed face boxes independently of a detector's inference framework."""

import argparse
import json
from pathlib import Path

from tinysr.common.io import write_json


def valid_box(box):
    """Use native QVGA half-open xywh coordinates."""
    import math

    return (
        len(box) == 4
        and all(math.isfinite(v) for v in box)
        and 0 <= box[0] < 320
        and 0 <= box[1] < 240
        and 0 < box[2] <= 320 - box[0]
        and 0 < box[3] <= 240 - box[1]
    )


def iou(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    width = max(0, min(a[0] + a[2], b[0] + b[2]) - x)
    height = max(0, min(a[1] + a[3], b[1] + b[3]) - y)
    intersection = width * height
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def score(truth, predictions, threshold=0.7, iou_threshold=0.5):
    """Greedily match score-sorted boxes; errors stay failures, not true negatives."""
    import math

    if not (0 <= threshold <= 1 and 0 < iou_threshold <= 1):
        raise ValueError("Invalid fixed screening thresholds")
    records = {r["scene_id"]: r for r in predictions}
    if len(records) != len(predictions) or len({r["scene_id"] for r in truth}) != len(
        truth
    ):
        raise ValueError("Duplicate scene IDs")
    if set(records) - {r["scene_id"] for r in truth}:
        raise ValueError("Predictions contain unknown scenes")
    tp = fp = fn = negative = negative_fp = 0
    failures = []
    sizes = {"under40": [0, 0], "40to79": [0, 0], "80plus": [0, 0]}
    for annotation in truth:
        sid = annotation["scene_id"]
        pred = records.get(sid)
        if (
            annotation.get("human_reviewed") is not True
            or pred is None
            or pred.get("error")
        ):
            failures.append(sid)
            continue
        if pred.get("box_semantics") != "face_bounds":
            raise ValueError(
                "Center heuristics cannot be scored as learned face bounds"
            )
        faces = annotation["face_boxes"]
        if not all(valid_box(box) for box in faces):
            raise ValueError("Invalid ground-truth box")
        boxes = pred["boxes"]
        if any(
            not valid_box(p["xywh"])
            or not math.isfinite(p["score"])
            or not 0 <= p["score"] <= 1
            for p in boxes
        ):
            raise ValueError("Invalid prediction")
        selected = sorted(
            (p for p in boxes if p["score"] >= threshold), key=lambda p: -p["score"]
        )
        used = set()
        for prediction in selected:
            candidates = [
                (iou(prediction["xywh"], box), index)
                for index, box in enumerate(faces)
                if index not in used
            ]
            best = max(candidates, default=(0, None))
            if best[0] >= iou_threshold:
                used.add(best[1])
                tp += 1
            else:
                fp += 1
        fn += len(faces) - len(used)
        if not faces:
            negative += 1
            negative_fp += bool(selected)
        for index, box in enumerate(faces):
            label = "under40" if box[2] < 40 else "40to79" if box[2] < 80 else "80plus"
            sizes[label][1] += 1
            sizes[label][0] += index in used
    return {
        "status": "incomplete" if failures else "complete",
        "failed_or_unreviewed_scenes": failures,
        "score_threshold": threshold,
        "iou_threshold": iou_threshold,
        "true_positive_boxes": tp,
        "false_positive_boxes": fp,
        "missed_face_boxes": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "no_face_frames_evaluated": negative,
        "false_positive_frame_rate": negative_fp / negative if negative else None,
        "recall_by_native_face_width": {
            k: {"matched": v[0], "total": v[1], "recall": v[0] / v[1] if v[1] else None}
            for k, v in sizes.items()
        },
    }
