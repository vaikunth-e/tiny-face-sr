"""Frozen split validation and portable preparation of aligned image pairs"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from tinysr.common.io import write_json


def validate_manifest(path):
    """Enforce unique image membership before reading any training pixels"""
    data = json.loads(Path(path).read_text())
    seen = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for row in data["pairs"]:
        identity = int(row["id"])
        if identity in seen or row["split"] not in counts:
            raise ValueError("Duplicate image ID or unknown split")
        seen.add(identity)
        counts[row["split"]] += 1
    if not counts["train"] or not counts["val"]:
        raise ValueError("Training and validation splits must both be present")
    return data, counts


def prepare(manifest, source, output):
    """Apply the historical Lanczos HR then bicubic LR preprocessing"""
    data, _ = validate_manifest(manifest)
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Use a new prepared dataset directory")
    index = {}
    for p in source.rglob("*.png"):
        if p.stem.isdigit():
            key = int(p.stem)
            if key in index:
                raise ValueError("Ambiguous source image ID")
            index[key] = p
    missing = [r["id"] for r in data["pairs"] if int(r["id"]) not in index]
    if missing:
        raise FileNotFoundError("Missing source images: " + str(len(missing)))
    output.mkdir(parents=True)
    rows = []
    for row in data["pairs"]:
        identifier = int(row["id"])
        split = row["split"]
        high = (
            Image.open(index[identifier])
            .convert("RGB")
            .resize((256, 256), Image.Resampling.LANCZOS)
        )
        low = high.resize((64, 64), Image.Resampling.BICUBIC)
        result = {"id": identifier, "split": split}
        for key, image in [("hr_path", high), ("lr_path", low)]:
            rel = Path(split) / key[:2] / (str(identifier).zfill(5) + ".png")
            (output / rel).parent.mkdir(parents=True, exist_ok=True)
            image.save(output / rel)
            result[key] = rel.as_posix()
        rows.append(result)
    write_json(output / "pairs.json", {"schema": "tinysr.pairs.v1", "pairs": rows})
    return output / "pairs.json"


class Pairs:
    """Read normalized RGB pairs from a current or historical local manifest"""

    def __init__(self, manifest, root, split):
        data, _ = validate_manifest(manifest)
        self.rows = [r for r in data["pairs"] if r["split"] == split]
        self.root = Path(root)
        if not self.rows:
            raise ValueError("Requested split is empty")
        if any(not all(k in r for k in ("hr_path", "lr_path")) for r in self.rows):
            raise ValueError("Split membership needs preparation before training")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        low = Image.open(self.root / row["lr_path"]).convert("RGB")
        high = Image.open(self.root / row["hr_path"]).convert("RGB")
        if low.size != (64, 64) or high.size != (256, 256):
            raise ValueError("Unexpected pair geometry")
        base = low.resize(high.size, Image.Resampling.BICUBIC)

        def tensor(image):
            return (
                torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy()).float()
                / 255
            )

        return tuple(tensor(x) for x in (low, base, high))
