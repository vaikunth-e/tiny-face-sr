"""Convert the supported residual baseline with audited output semantics"""

from pathlib import Path
import numpy as np
from tinysr.common.io import sha256, write_json


def export(checkpoint, output, calibration=None):
    """Require real representative input tiles for any INT8 export"""
    import torch
    import tensorflow as tf
    from tinysr.models.residual import TinySR
    from .tensorflow import tensorflow_core, interpreter, audit

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state.get("model_config") or state.get("frozen_identity", {}).get(
        "model_config"
    )
    if (
        not cfg
        or cfg["variant"] != "residual"
        or cfg["blocks"] != 2
        or cfg["scale"] != 4
    ):
        raise ValueError("Only the historical residual export topology is supported")
    model = TinySR(**cfg).eval()
    model.load_state_dict(state["model_state"], strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new export directory")
    samples = None
    if calibration is not None:
        samples = np.load(calibration, allow_pickle=False)
        if (
            samples.ndim != 4
            or samples.shape[1:] != (64, 64, 3)
            or len(samples) < 1
            or not np.isfinite(samples).all()
            or samples.min() < 0
            or samples.max() > 1
        ):
            raise ValueError("Calibration must contain normalized RGB [N,64,64,3]")
    output.mkdir(parents=True)
    core = tensorflow_core(model)
    concrete = core.infer.get_concrete_function()
    kinds = ["fp32"] + (["int8"] if samples is not None else [])
    reports = {}
    for kind in kinds:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], core)
        if kind == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: (
                [x[None].astype(np.float32)] for x in samples
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        raw = converter.convert()
        path = output / ("residual_" + kind + ".tflite")
        path.write_bytes(raw)
        reports[kind] = {
            "sha256": sha256(path),
            "bytes": len(raw),
            **audit(interpreter(raw), kind == "int8"),
        }
    write_json(
        output / "conversion.json",
        {
            "schema": "tinysr.conversion.v1",
            "source_sha256": sha256(checkpoint),
            "semantics": "normalized_rgb_residual",
            "models": reports,
            "hardware_verified": False,
        },
    )
    return output
