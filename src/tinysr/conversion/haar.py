"""Convert compatible OpenCV Haar cascades to OpenMV v4.5.9 binary layout

Derived independently from the documented v4.5.9 tools/haar2c.py field layout
The on-camera Haar engine supports stump classifiers, not general tree classifiers
The script validates the XML rather than silently truncating unsupported trees
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


class UnsupportedCascade(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixed(text: str, multiplier: int, limits=(-32768, 32767)) -> int:
    # this reproduces the upstream int(float(x)*factor) truncation semantics
    result = int(float(text) * multiplier)
    if not limits[0] <= result <= limits[1]:
        raise UnsupportedCascade("Fixed-point field is outside the OpenMV int16 range")
    return result


def convert_xml(xml: bytes) -> tuple[bytes, dict]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as ex:
        raise UnsupportedCascade("Invalid XML") from ex
    cascade = root.find("cascade")
    if (
        cascade is None
        or cascade.find("stages") is None
        or cascade.find("features") is None
    ):
        raise UnsupportedCascade("Expected OpenCV new-format cascade XML")
    if cascade.findtext("featureType") != "HAAR":
        raise UnsupportedCascade("Only Haar featureType is supported")
    if cascade.findtext("stageType") != "BOOST":
        raise UnsupportedCascade("Only BOOST stageType is supported")
    width, height = int(cascade.findtext("width")), int(cascade.findtext("height"))
    if not 1 <= width <= 255 or not 1 <= height <= 255:
        raise UnsupportedCascade("Invalid integral-image feature window size")
    stages = cascade.find("stages").findall("./_")
    if len(stages) != int(cascade.findtext("stageNum")) or not stages:
        raise UnsupportedCascade("Inconsistent stage count")
    stage_counts = []
    stage_thresholds = []
    nodes = []
    leaves = []
    for stage in stages:
        weak = stage.find("weakClassifiers")
        if weak is None:
            raise UnsupportedCascade("Missing weak classifiers")
        weak_items = weak.findall("./_")
        n = int(stage.findtext("maxWeakCount"))
        if n != len(weak_items) or not 1 <= n <= 255:
            raise UnsupportedCascade("Invalid stage weak-classifier count")
        stage_counts.append(n)
        stage_thresholds.append(_fixed(stage.findtext("stageThreshold"), 256))
        for w in weak_items:
            ni = w.findtext("internalNodes", "").split()
            lv = w.findtext("leafValues", "").split()
            # multi-node decision trees, including alt2, cannot be represented
            if len(ni) != 4 or len(lv) != 2 or ni[:2] != ["0", "-1"]:
                raise UnsupportedCascade(
                    "Weak classifier is not a two-leaf stump (OpenMV v4.5.9 format cannot represent it)"
                )
            nodes.append((int(ni[2]), _fixed(ni[3], 4096)))
            leaves.append((_fixed(lv[0], 256), _fixed(lv[1], 256)))
    features = cascade.find("features").findall("./_")
    feature_rectangles = []
    for f in features:
        rect_items = f.find("rects")
        if rect_items is None:
            raise UnsupportedCascade("Feature has no rectangles")
        rects = []
        for r in rect_items.findall("./_"):
            values = (r.text or "").strip().split()
            if len(values) != 5:
                raise UnsupportedCascade("Malformed rectangle")
            x, y, w, h = (int(v) for v in values[:4])
            weight_float = float(values[4])
            if weight_float != int(weight_float):
                raise UnsupportedCascade(
                    "Rectangle weight not exactly representable as int8"
                )
            weight = int(weight_float)
            if not (
                0 <= x < width
                and 0 <= y < height
                and 1 <= w <= width - x
                and 1 <= h <= height - y
                and -128 <= weight <= 127
            ):
                raise UnsupportedCascade("Rectangle coordinates/weight out of range")
            rects.append((x, y, w, h, weight))
        if not 1 <= len(rects) <= 255:
            raise UnsupportedCascade("Invalid number of rectangles for feature")
        feature_rectangles.append(rects)
    referenced = []
    for feature_id, _ in nodes:
        if not 0 <= feature_id < len(feature_rectangles):
            raise UnsupportedCascade("Classifier references a nonexistent feature")
        referenced.append(feature_rectangles[feature_id])
    if len(nodes) != sum(stage_counts):
        raise UnsupportedCascade("Weak classifier count mismatch")
    # exact original OpenMV 4.5.9 binary field order, native little-endian RT1062
    parts = [struct.pack("<iii", width, height, len(stages))]
    parts.append(bytes(stage_counts))
    parts.append(struct.pack("<" + "h" * len(stages), *stage_thresholds))
    parts.append(
        struct.pack("<" + "h" * len(nodes), *(threshold for _, threshold in nodes))
    )
    parts.append(struct.pack("<" + "h" * len(nodes), *(a for a, _ in leaves)))
    parts.append(struct.pack("<" + "h" * len(nodes), *(b for _, b in leaves)))
    parts.append(bytes(len(rects) for rects in referenced))
    for rects in referenced:
        for x, y, w, h, weight in rects:
            parts.append(struct.pack("<b", weight))
    for rects in referenced:
        for x, y, w, h, weight in rects:
            parts.append(struct.pack("<BBBB", x, y, w, h))
    output = b"".join(parts)
    num_rectangles = sum(map(len, referenced))
    expected_size = 12 + len(stages) * 3 + len(nodes) * 7 + num_rectangles * 5
    if len(output) != expected_size:
        raise AssertionError("Cascade binary length mismatch")
    meta = {
        "window": [width, height],
        "stages": len(stages),
        "weak_classifiers": len(nodes),
        "referenced_rectangles": num_rectangles,
        "bytes": len(output),
        "sha256": digest(output),
        "xml_sha256": digest(xml),
        "weak_classifier_type": "two_leaf_stump",
    }
    return output, meta


def inspect_binary(blob: bytes) -> dict:
    if len(blob) < 15:
        raise ValueError("Truncated cascade")
    width, height, nstages = struct.unpack_from("<iii", blob)
    if not (0 < width <= 255 and 0 < height <= 255 and 0 < nstages <= 255):
        raise ValueError("Bad cascade header")
    counts = blob[12 : 12 + nstages]
    if len(counts) != nstages or any(x == 0 for x in counts):
        raise ValueError("Truncated stage counts")
    n = sum(counts)
    counts_offset = 12 + nstages + 2 * nstages + 2 * n * 3
    rect_counts = blob[counts_offset : counts_offset + n]
    if len(rect_counts) != n or any(x == 0 for x in rect_counts):
        raise ValueError("Truncated rectangle counts")
    nr = sum(rect_counts)
    if len(blob) != counts_offset + n + nr * 5:
        raise ValueError("Invalid binary size")
    return {
        "window": [width, height],
        "stages": nstages,
        "weak_classifiers": n,
        "referenced_rectangles": nr,
        "bytes": len(blob),
        "sha256": digest(blob),
    }


def build(source_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError("Preserve previous cascade exports")
    output_dir.mkdir(parents=True)
    manifest = {
        "experiment": "TinySR_v5_5_compatible_Haar",
        "converter": "convert_haar.py",
        "upstream_layout": "OpenMV v4.5.9 tools/haar2c.py cascade_binary()",
        "models": {},
    }
    for stem in ("haarcascade_frontalface_default", "haarcascade_frontalface_alt"):
        src = source_dir / (stem + ".xml")
        blob, meta = convert_xml(src.read_bytes())
        name = stem + ".cascade"
        (output_dir / name).write_bytes(blob)
        if inspect_binary((output_dir / name).read_bytes())["sha256"] != meta["sha256"]:
            raise RuntimeError("Read-back binary identity mismatch")
        manifest["models"][stem] = {"filename": name, **meta}
    alt2 = source_dir / "haarcascade_frontalface_alt2.xml"
    if alt2.exists():
        try:
            convert_xml(alt2.read_bytes())
        except UnsupportedCascade as e:
            manifest["excluded_alt2"] = {
                "xml_sha256": digest(alt2.read_bytes()),
                "reason": str(e),
                "note": "Not valid to truncate tree classifier to stump",
            }
        else:
            raise AssertionError("alt2 was expected to use unsupported tree nodes")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
