"""Model identity and cross-runtime parity checks that require no face dataset."""

import numpy as np
from pathlib import Path

from tinysr.models.runtime import TorchResidual, LiteResidual
from tinysr.models.registry import model_records, selected_artifact
from tinysr.common.io import write_json
from tinysr.common.device import environment
from tinysr.evaluation.compose import cost


def verify(output):
    """Exercise actual recovered weights with deterministic non-face witnesses."""
    import torch

    torch.set_num_threads(1)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new verification output")
    rng = np.random.default_rng(1337)
    witnesses = [
        np.zeros((64, 64, 3), np.uint8),
        np.full((64, 64, 3), 127, np.uint8),
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
    ]
    report = {
        "evidence": "fresh desktop inference on synthetic non-face witnesses",
        "environment": environment(),
        "models": [],
        "quality_result": False,
    }
    for record in model_records():
        pytorch = TorchResidual(
            selected_artifact(record, "best.pt"), record["channels"]
        )
        floating = LiteResidual(
            selected_artifact(record, "residual_fp32.tflite"), integer=False
        )
        integer = LiteResidual(selected_artifact(record, "residual_int8.tflite"))
        errors = []
        for witness in witnesses:
            a, b = pytorch.predict(witness), floating.predict(witness)
            error = float(np.max(np.abs(a - b)))
            if not np.allclose(a, b, atol=1e-5, rtol=1e-4):
                raise AssertionError("Float parity failed: " + record["id"])
            q, raw = integer.invoke(witness)
            if not np.isfinite(q).all():
                raise AssertionError("Nonfinite INT8 output")
            errors.append(
                {
                    "torch_float_lite_max_abs_residual_error": error,
                    "int8_float_max_abs_residual_difference": float(
                        np.max(np.abs(a - q))
                    ),
                    "int8_outputs_at_dtype_limits": int(
                        np.count_nonzero((raw == -128) | (raw == 127))
                    ),
                    "float_residual_min": float(a.min()),
                    "float_residual_max": float(a.max()),
                    "interpretation": "stress witness; finite execution is not an INT8 quality gate",
                }
            )
        row = {
            "model": record["id"],
            "float_parity_passed": True,
            "witnesses": errors,
            "torch": pytorch.metadata,
            "float_tflite": floating.metadata,
            "int8_tflite": integer.metadata,
            **cost(record["channels"]),
        }
        report["models"].append(row)
    report["status"] = "passed"
    write_json(output, report)
    print("Verified three recovered checkpoints and six TFLite exports")
    return report
