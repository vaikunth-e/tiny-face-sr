"""Immutable baseline identity records with an explicit asset-root override"""

import json
import os
from pathlib import Path
from tinysr.common.io import sha256


def project_root():
    """Locate source checkout assets or a separately configured model root"""
    configured = os.environ.get("TINYSR_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "models/registry.json").is_file():
            return parent
    raise RuntimeError(
        "Set TINYSR_ROOT to the checkout containing models/registry.json"
    )


def model_records():
    return json.loads((project_root() / "models/registry.json").read_text())["models"]


def selected_artifact(record, filename):
    """Reject changed artifacts rather than selecting weights by approximate name"""
    item = record["artifacts"][filename]
    path = project_root() / item["path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError("Model identity mismatch: " + record["id"] + "/" + filename)
    return path


def audit():
    """Validate hashes without importing neural runtimes or disclosing local paths"""
    return [
        {
            "id": r["id"],
            "files_verified": len(
                [selected_artifact(r, name) for name in r["artifacts"]]
            ),
            "semantics": r["semantics"],
        }
        for r in model_records()
    ]
