"""Keep camera receipts local and export only numeric stage observations"""

from pathlib import Path
import zipfile
import json
from tinysr.common.io import write_json
from .session import local_comparison


def save_results(output, result, images, script):
    """Keep finished images, raw logs and source configuration in local-only storage"""
    output = Path(output)
    local = output / "local_only"
    local.mkdir(parents=True, exist_ok=False)
    write_json(local / "receipt.json", result)
    (local / "camera_program.py").write_text(script, encoding="utf-8")
    if result.get("host_state") == "completed":
        for key, name in [
            ("BASE", "camera_bicubic_4x.jpg"),
            ("NEURAL", "camera_final_4x.jpg"),
        ]:
            (local / name).write_bytes(images[key])
        local_comparison(
            images, local, neural_applied=result["device"].get("neural_applied", False)
        )
    device = result.get("device", {})
    stages = []
    for stage in device.get("stages", []):
        safe = {
            k: v
            for k, v in stage.items()
            if k
            in {
                "milliseconds",
                "frame_bytes",
                "jpeg_bytes",
                "model_bytes",
                "arena_bytes",
                "neural_tiles",
                "candidates",
            }
            and type(v) in (int, float)
        }
        # stage labels are code-defined and never accept raw camera text
        if safe:
            stages.append(safe)
    report = {
        "schema": "tinysr.camera_metrics.v1",
        "completed": result.get("host_state") == "completed",
        "neural_applied": bool(device.get("neural_applied", False)),
        "tiles": int(device.get("tiles", 0)),
        "stages": stages,
        "neural": {
            k: v
            for k, v in device.get("neural", {}).items()
            if k
            in {
                "inference_ms",
                "composition_ms",
                "changed_pixels",
                "changed_channels",
                "sum_abs_delta",
            }
            and type(v) in (int, float)
        },
    }
    with zipfile.ZipFile(output / "metrics_only.zip", "x", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metrics.json", json.dumps(report, indent=2))
    return output
