"""Exact PixelShuffle channel mapping and residual export core"""

import numpy as np


def depth_to_space_filter_order(scale=4, colors=3):
    """PyTorch head: [color,dy,dx]; TensorFlow head: [dy,dx,color]."""
    return (
        np.arange(colors * scale * scale)
        .reshape(colors, scale, scale)
        .transpose(1, 2, 0)
        .reshape(-1)
    )


def tensorflow_core(reference):
    """Export only the learned residual; external reconstruction stays unchanged."""
    import tensorflow as tf
    import torch

    convolutions = [
        layer for layer in reference.network if isinstance(layer, torch.nn.Conv2d)
    ]
    if len(convolutions) != 4:
        raise RuntimeError(
            "Expected three feature convolutions and one output convolution."
        )
    kernels, biases = [], []
    for index, layer in enumerate(convolutions):
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        if index == len(convolutions) - 1:
            order = depth_to_space_filter_order()
            weight, bias = weight[order], bias[order]
        kernels.append(tf.constant(weight.transpose(2, 3, 1, 0), dtype=tf.float32))
        biases.append(tf.constant(bias, dtype=tf.float32))

    class ResidualCore(tf.Module):
        def __init__(self):
            super().__init__()
            self.kernels = kernels
            self.biases = biases
            self.infer = tf.function(
                self.forward,
                input_signature=[
                    tf.TensorSpec([1, 64, 64, 3], tf.float32, name="lr_rgb")
                ],
                autograph=False,
            )

        def forward(self, low):
            value = low
            for index, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):
                value = tf.nn.bias_add(
                    tf.nn.conv2d(value, kernel, strides=1, padding="SAME"), bias
                )
                if index < 3:
                    value = tf.nn.relu(value)
            return tf.identity(
                tf.nn.depth_to_space(value, 4, data_format="NHWC"), name="residual_rgb"
            )

    return ResidualCore()


def quantize(array, detail):
    """Apply affine tensor quantization, then clip to the integer dtype range."""
    scale, zero = detail["quantization"]
    if scale <= 0:
        raise RuntimeError("Missing positive quantization scale.")
    limits = np.iinfo(detail["dtype"])
    raw = np.rint(array / scale + zero)
    return np.clip(raw, limits.min, limits.max).astype(detail["dtype"])


def interpreter(model_bytes):
    """Use builtin desktop kernels without default delegates for a clear audit."""
    import tensorflow as tf

    runtime = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    runtime.allocate_tensors()
    return runtime


def invoke(runtime, low):
    """Return dequantized residual and the actual output tensor for diagnostics."""
    inp = runtime.get_input_details()[0]
    out = runtime.get_output_details()[0]
    value = (
        quantize(low, inp)
        if np.issubdtype(inp["dtype"], np.integer)
        else low.astype(inp["dtype"])
    )
    runtime.set_tensor(inp["index"], value)
    runtime.invoke()
    raw = runtime.get_tensor(out["index"])
    if np.issubdtype(out["dtype"], np.integer):
        scale, zero = out["quantization"]
        if scale <= 0:
            raise RuntimeError("Invalid output quantization scale.")
        residual = (raw.astype(np.float32) - zero) * scale
    else:
        residual = raw.astype(np.float32)
    return residual, raw


def audit(runtime, strict_int8=False):
    """Inspect actual tensor dtypes and operators; do not assume target support."""

    def tensor(detail):
        return {
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "dtype": np.dtype(detail["dtype"]).name,
            "scale": float(detail["quantization"][0]),
            "zero_point": int(detail["quantization"][1]),
        }

    inputs = [tensor(d) for d in runtime.get_input_details()]
    outputs = [tensor(d) for d in runtime.get_output_details()]
    tensors = [tensor(d) for d in runtime.get_tensor_details()]
    # this private interpreter diagnostic is recorded with the TensorFlow version
    ops = [d["op_name"] for d in runtime._get_ops_details()]
    if strict_int8:
        if inputs[0]["dtype"] != "int8" or outputs[0]["dtype"] != "int8":
            raise RuntimeError("Full INT8 input/output requirement failed.")
        if any(t["dtype"].startswith("float") for t in tensors):
            raise RuntimeError("A floating-point tensor remains in the integer model.")
        if any(
            op not in {"CONV_2D", "DEPTH_TO_SPACE", "RESHAPE", "RELU"} for op in ops
        ):
            raise RuntimeError(f"Unexpected operator in neural core: {ops}")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "tensors": tensors,
        "operators": ops,
        "strict_integer_core": strict_int8,
        "target_compatibility": "NOT YET VERIFIED ON OPENMV/TFLITE MICRO",
    }
