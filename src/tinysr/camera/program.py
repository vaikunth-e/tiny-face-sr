"""Assemble maintained MicroPython source into one temporary camera program"""

from importlib.resources import files
from tinysr.models.registry import model_records


def board_program(
    *,
    mode="auto",
    manual=None,
    tx_bytes=96,
    tx_pause_ms=35,
    jpeg_quality=90,
    fomo_threshold=0.40,
    head_size=140,
    max_tiles=16,
    max_jpeg_bytes=3_000_000,
    model_path="/tinysr_rt1062/c16/residual_int8.tflite",
):
    """Validate the camera contract before sending any executable code"""
    if mode not in ("auto", "manual", "bicubic"):
        raise ValueError("Camera mode must be auto, manual, or bicubic")
    if manual is not None:
        manual = tuple(int(v) for v in manual)
        if len(manual) != 4 or min(manual[2:]) <= 0:
            raise ValueError("Manual region must have positive dimensions")
        x, y, w, h = manual
        if min(x, y) < 0 or x + w > 320 or y + h > 240:
            raise ValueError("Manual region must fit QVGA")
        mode = "manual"
    if mode == "manual" and manual is None:
        raise ValueError("Manual mode requires an explicit ROI")
    if not 32 <= tx_bytes <= 192 or not 10 <= tx_pause_ms <= 100:
        raise ValueError("Transfer chunk size or pacing is outside verified bounds")
    if not 40 <= jpeg_quality <= 95 or not 0.05 <= fomo_threshold <= 0.95:
        raise ValueError("Invalid JPEG quality or detector threshold")
    if not 32 <= head_size <= 180 or not 1 <= max_tiles <= 30:
        raise ValueError("Invalid ROI size or tile budget")
    if not 1000 <= max_jpeg_bytes <= 5_000_000:
        raise ValueError("Invalid JPEG size limit")
    if not isinstance(model_path, str) or not model_path or "\n" in model_path:
        raise ValueError("Invalid on-device model path")
    c16 = next(r for r in model_records() if r["id"] == "C16")
    constants = {
        "TAG": "TINYSR",
        "W": 320,
        "H": 240,
        "SCALE": 4,
        "TILE": 64,
        "HALO": 3,
        "STEP": 58,
        "OUT_W": 1280,
        "OUT_H": 960,
        "TX_BYTES": tx_bytes,
        "TX_PAUSE": tx_pause_ms,
        "QUALITY": jpeg_quality,
        "MAX_JPEG_BYTES": max_jpeg_bytes,
        "MANUAL": manual,
        "MODE": mode,
        "FOMO_THRESHOLD": fomo_threshold,
        "HEAD_SIZE": head_size,
        "MAX_TILES": max_tiles,
        "MODEL_BYTES": 11152,
        "MODEL_PATH": model_path,
        "MODEL_SHA256": c16["artifacts"]["residual_int8.tflite"]["sha256"],
    }
    script = "import gc, image, sensor, os, time, ujson, uhashlib, ubinascii, sys, ml, math\nfrom ml.utils import NMS\n"
    script += "\n".join(k + "=" + repr(v) for k, v in constants.items()) + "\n"
    script += "report={'experiment':'camera_fullframe_v1','state':'started','stages':[], 'roi':MANUAL,'roi_mode':MODE,'tiles':0,'errors':[], 'detector':None,'neural_applied':False,'jpeg_quality':QUALITY, 'color_space':'RGB565','residual_scale':'normalized_times_255'}\n"
    # execute shared geometry definitions in the same MicroPython namespace
    script += files("tinysr.common").joinpath("geometry.py").read_text() + "\n"
    script += 'def tiles_for(roi):\n    tiles=plan(roi, max_tiles=1000)\n    if len(tiles)>MAX_TILES:\n        raise RuntimeError("selected region exceeds neural tile budget")\n    return tiles\n\nclip_roi=clamp\n'
    root = files("tinysr.device.openmv")
    for name in ("transport", "detection", "superresolution", "main"):
        script += "\n" + root.joinpath(name + ".py").read_text() + "\n"
    compile(script, "camera_program", "exec")
    return script
