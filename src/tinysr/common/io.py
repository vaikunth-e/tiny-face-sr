"""Atomic records and immutable artifact hashing"""

import hashlib
import json
from pathlib import Path


def sha256(path):
    """Hash model and local input files without loading large files into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    """Write strict JSON atomically so incomplete reports are not mistaken for results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
