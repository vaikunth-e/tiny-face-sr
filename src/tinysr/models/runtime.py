"""Strict checkpoint and TFLite residual adapters"""

from pathlib import Path
import numpy as np
from tinysr.common.io import sha256
from tinysr.models.registry import selected_artifact, model_records


class TorchResidual:
    """Load actual model_state checkpoints through the original strict architecture."""

    def __init__(self, path, channels, device="cpu"):
        import torch
        from .residual import TinySR

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a checkpoint dictionary")
        top = checkpoint.get("model_config")
        frozen = checkpoint.get("frozen_identity", {}).get("model_config")
        if top and frozen and top != frozen:
            raise ValueError("Conflicting checkpoint architecture metadata")
        expected = {
            "variant": "residual",
            "channels": channels,
            "blocks": 2,
            "scale": 4,
        }
        if (top or frozen) != expected:
            raise ValueError(
                "Checkpoint does not describe the baseline residual architecture"
            )
        state = checkpoint.get("model_state")
        if state is None:
            raise ValueError(
                "Missing model_state; arbitrary state-dict layouts are not guessed"
            )
        self.model = TinySR(**expected)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.metadata = {
            "backend": "pytorch_fp32",
            "model_sha256": sha256(path),
            "file_bytes": Path(path).stat().st_size,
            "channels": channels,
            "checkpoint_step": checkpoint.get("step"),
        }

    def predict(self, rgb):
        """Return signed normalized residuals, never reconstructed RGB pixels."""
        import torch

        a = np.asarray(rgb, dtype=np.float32)
        if a.shape != (64, 64, 3):
            raise ValueError("Expected native 64x64 RGB tile")
        x = torch.from_numpy((a / 255).transpose(2, 0, 1).copy())[None].to(self.device)
        with torch.inference_mode():
            y = self.model.network(x)
        return y[0].permute(1, 2, 0).cpu().numpy().copy()


def interpreter(path):
    """Use explicit builtin kernels without a default delegate for reproducible audits."""
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter, OpResolverType
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
    runtime = Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    runtime.allocate_tensors()
    return runtime


class LiteResidual:
    """Read each exported model's own affine scales and zero points."""

    def __init__(self, path, integer=True):
        self.runtime = interpreter(path)
        ins, outs = self.runtime.get_input_details(), self.runtime.get_output_details()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError("Expected one input and one output")
        self.inp, self.out = ins[0], outs[0]
        dtype = np.int8 if integer else np.float32
        for detail, shape in ((self.inp, (1, 64, 64, 3)), (self.out, (1, 256, 256, 3))):
            if tuple(detail["shape"]) != shape or detail["dtype"] != dtype:
                raise ValueError("Residual tensor contract mismatch")
            if integer and detail["quantization"][0] <= 0:
                raise ValueError("Missing positive quantization scale")
        self.integer = integer
        self.metadata = {
            "backend": "litert_int8" if integer else "litert_fp32",
            "model_sha256": sha256(path),
            "file_bytes": Path(path).stat().st_size,
            "input": self.detail(self.inp),
            "output": self.detail(self.out),
            "operators": [op["op_name"] for op in self.runtime._get_ops_details()],
            "audit_api": "private _get_ops_details; version recorded in environment",
        }

    @staticmethod
    def detail(d):
        return {
            "shape": d["shape"].tolist(),
            "dtype": np.dtype(d["dtype"]).name,
            "scale": float(d["quantization"][0]),
            "zero_point": int(d["quantization"][1]),
        }

    def invoke(self, rgb):
        """Return the dequantized residual and raw output for parity measurements."""
        a = np.asarray(rgb, dtype=np.float32)
        if a.shape != (64, 64, 3):
            raise ValueError("Expected native 64x64 RGB tile")
        a = a[None] / 255
        if self.integer:
            scale, zero = self.inp["quantization"]
            a = np.clip(np.rint(a / scale + zero), -128, 127).astype(np.int8)
        self.runtime.set_tensor(self.inp["index"], a)
        self.runtime.invoke()
        raw = self.runtime.get_tensor(self.out["index"])[0]
        y = raw.astype(np.float32)
        if self.integer:
            scale, zero = self.out["quantization"]
            y = (y - zero) * scale
        return y, raw

    def predict(self, rgb):
        return self.invoke(rgb)[0]
