"""Resume reproduces uninterrupted learning and rejects changed execution identity"""

import json
import numpy as np
import pytest
from PIL import Image
from tinysr.common.io import write_json
from tinysr.training.engine import train


def dataset(root):
    rows = []
    rng = np.random.default_rng(3)
    for i, split in enumerate(["train", "train", "val"]):
        row = {
            "id": i,
            "split": split,
            "lr_path": str(i) + "_lr.png",
            "hr_path": str(i) + "_hr.png",
        }
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            root / row["lr_path"]
        )
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(
            root / row["hr_path"]
        )
        rows.append(row)
    write_json(root / "pairs.json", {"pairs": rows})
    return root / "pairs.json"


def test_exact_cpu_resume_and_identity(tmp_path):
    import torch

    manifest = dataset(tmp_path)
    cfg = {
        "variant": "residual",
        "channels": 4,
        "blocks": 2,
        "scale": 4,
        "steps": 4,
        "batch_size": 1,
        "validation_every": 2,
        "seed": 17,
        "lr_schedule": [{"start": 1, "learning_rate": 0.001}],
    }
    train(cfg, manifest, tmp_path, tmp_path / "whole", device="cpu")
    train(cfg, manifest, tmp_path, tmp_path / "resume", device="cpu", stop_at=2)
    (tmp_path / "resume/best.pt").unlink()
    train(cfg, manifest, tmp_path, tmp_path / "resume", device="cpu")
    a = torch.load(tmp_path / "whole/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    for key in a["model_state"]:
        assert torch.equal(a["model_state"][key], b["model_state"][key])
    cfg["batch_size"] = 2
    with pytest.raises(ValueError, match="identity"):
        train(cfg, manifest, tmp_path, tmp_path / "resume", device="cpu")


def test_explicit_unavailable_cuda_fails(monkeypatch):
    import torch
    from tinysr.common.device import select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        select_device("cuda")
