"""Camera output boundary, identity checks and private receipt separation"""

import io, json, zipfile
from PIL import Image
import pytest
from tinysr.camera.program import board_program
from tinysr.camera.results import save_results
from tinysr.camera.session import inspect_jpeg


def test_finished_jpeg_and_sanitized_receipt(tmp_path):
    f = io.BytesIO()
    Image.new("RGB", (1280, 960)).save(f, "JPEG")
    blob = f.getvalue()
    assert inspect_jpeg(blob)["width"] == 1280
    result = {
        "host_state": "completed",
        "camera_console_tail": "PRIVATE_MARKER",
        "device": {
            "state": "completed",
            "neural_applied": False,
            "tiles": 0,
            "cwd": "PRIVATE_MARKER",
            "stages": [],
            "neural": {},
        },
    }
    out = save_results(
        tmp_path / "run", result, {"BASE": blob, "NEURAL": blob}, "# local program"
    )
    assert (out / "local_only/camera_final_4x.jpg").read_bytes() == blob
    with zipfile.ZipFile(out / "metrics_only.zip") as z:
        assert z.namelist() == ["metrics.json"]
        assert b"PRIVATE_MARKER" not in z.read("metrics.json")
    with pytest.raises(ValueError):
        inspect_jpeg(blob[:-2])


def test_manual_bounds_and_model_identity_gate():
    with pytest.raises(ValueError):
        board_program(mode="manual")
    with pytest.raises(ValueError):
        board_program(manual=(300, 0, 100, 100))
    script = board_program(manual=(96, 56, 128, 128))
    assert "MODEL_SHA256=" in script
    assert "model_digest != MODEL_SHA256" in script
    assert "MODE='manual'" in script
