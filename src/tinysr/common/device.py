"""Local capability discovery without machine inventory publication"""


def select_device(request="auto"):
    """Prefer CUDA for auto inference and fail on unavailable explicit requests"""
    import torch

    if request == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu, cuda, and auto")
    if device.type == "cuda":
        index = device.index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(
                "Requested CUDA device is unavailable; CPU fallback is disabled"
            )
    return str(device)


def environment():
    """Return portable capabilities rather than personal hardware or driver details"""
    try:
        import torch

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_available": True,
        }
    except ImportError:
        return {
            "cuda_available": False,
            "cpu_available": True,
            "torch_available": False,
        }
