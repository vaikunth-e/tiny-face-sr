"""Configurable baseline training with an exact saved sampler and optimizer cursor"""

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import numpy as np
from PIL import Image
from tinysr.common.io import sha256, write_json
from tinysr.common.device import select_device
from tinysr.data.pairs import Pairs
from tinysr.evaluation.compose import paired_metrics


def atomic_save(path, state):
    """Publish a checkpoint only after serialization has completed"""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class BatchOrder:
    """Save shuffle state and cursor so resumed batches match uninterrupted training"""

    def __init__(self, length, seed):
        import torch

        self.length = length
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(length, generator=self.generator)
        self.cursor = 0

    def next(self, size):
        import torch

        if size > self.length or size < 1:
            raise ValueError("Batch size exceeds dataset")
        if self.cursor + size > self.length:
            self.order = torch.randperm(self.length, generator=self.generator)
            self.cursor = 0
        values = self.order[self.cursor : self.cursor + size].tolist()
        self.cursor += size
        return values

    def state_dict(self):
        return {
            "length": self.length,
            "order": self.order,
            "cursor": self.cursor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if state["length"] != self.length:
            raise ValueError("Dataset length changed")
        self.order = state["order"]
        self.cursor = state["cursor"]
        self.generator.set_state(state["generator_state"])


def evaluate(model, dataset, device, batch_size=4):
    """Average per-image BT.601 luma scores without evaluating the test partition"""
    import torch

    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = [
                dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
            ]
            tensors = [
                torch.stack([row[k] for row in batch]).to(device) for k in range(3)
            ]
            prediction = (
                model(tensors[0], tensors[1]).clamp(0, 1).mul(255).round().byte().cpu()
            )
            truth = tensors[2].mul(255).round().byte().cpu()
            for a, b in zip(truth, prediction):
                scores.append(
                    paired_metrics(
                        Image.fromarray(a.permute(1, 2, 0).numpy()),
                        Image.fromarray(b.permute(1, 2, 0).numpy()),
                    )
                )
    # exact matches have infinite PSNR and cannot be represented as finite JSON
    psnrs = [s["psnr_y_db"] if s["psnr_y_db"] is not None else 100.0 for s in scores]
    return {
        "psnr_y_db": float(np.mean(psnrs)),
        "ssim_y": float(np.mean([s["ssim_y"] for s in scores])),
        "perfect_matches": sum(s["perfect_match"] for s in scores),
        "perfect_psnr_cap_db": 100,
    }


def train(config, manifest, data_root, output, device="cuda", stop_at=None):
    """Use explicit CPU opt-in and refuse resume after data, code, or execution changes"""
    import torch
    from tinysr.models.residual import TinySR

    cfg = (
        json.loads(Path(config).read_text())
        if isinstance(config, (str, Path))
        else dict(config)
    )
    required = (
        "channels",
        "blocks",
        "variant",
        "scale",
        "steps",
        "batch_size",
        "validation_every",
        "seed",
        "lr_schedule",
    )
    if any(k not in cfg for k in required):
        raise ValueError("Incomplete training configuration")
    if (
        cfg["scale"] != 4
        or min(cfg["steps"], cfg["batch_size"], cfg["validation_every"]) < 1
    ):
        raise ValueError("Invalid training geometry or budget")
    schedule = cfg["lr_schedule"]
    if (
        not schedule
        or schedule[0]["start"] != 1
        or any(a["start"] >= b["start"] for a, b in zip(schedule, schedule[1:]))
        or any(s["learning_rate"] <= 0 for s in schedule)
    ):
        raise ValueError("Invalid learning rate schedule")
    device = select_device(device)
    training = Pairs(manifest, data_root, "train")
    validation = Pairs(manifest, data_root, "val")
    if len(training) < cfg["batch_size"]:
        raise ValueError("Training split is smaller than one batch")
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model_cfg = {k: cfg[k] for k in ("variant", "channels", "blocks", "scale")}
    model = TinySR(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    sampler = BatchOrder(len(training), cfg["seed"] + 17)
    code_hashes = {
        str(p.relative_to(Path(__file__).parents[1])): sha256(p)
        for p in sorted(Path(__file__).parents[1].rglob("*.py"))
    }
    identity = {
        "config": cfg,
        "manifest_sha256": sha256(manifest),
        "device": device,
        "code_sha256": code_hashes,
        "torch": str(torch.__version__),
    }
    output = Path(output)
    last = output / "last.pt"
    completed = 0
    history = []
    best = -float("inf")
    best_weights = None
    best_step = None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume identity changed")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        sampler.load_state_dict(state["batch_order"])
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed = state["step"]
        history = state["history"]
        best = state["best_psnr"]
        best_weights = state["best_model_state"]
        best_step = state["best_step"]
        if not (output / "best.pt").exists():
            atomic_save(
                output / "best.pt",
                {
                    "model_config": model_cfg,
                    "model_state": best_weights,
                    "step": best_step,
                },
            )
    elif output.exists():
        raise FileExistsError("Existing output has no resumable checkpoint")
    else:
        output.mkdir(parents=True)
    target = cfg["steps"] if stop_at is None else stop_at
    if not 1 <= target <= cfg["steps"]:
        raise ValueError("Stop update is outside training budget")
    for step in range(completed + 1, target + 1):
        rate = [s["learning_rate"] for s in schedule if s["start"] <= step][-1]
        for group in optimizer.param_groups:
            group["lr"] = rate
        batch = [training[i] for i in sampler.next(cfg["batch_size"])]
        low, base, high = [
            torch.stack([r[k] for r in batch]).to(device) for k in range(3)
        ]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(low, base), high)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % cfg["validation_every"] and step != target:
            continue
        metrics = evaluate(model, validation, device, cfg["batch_size"])
        history.append({"step": step, "train_l1": float(loss.detach()), **metrics})
        if metrics["psnr_y_db"] > best:
            best = metrics["psnr_y_db"]
            best_step = step
            best_weights = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            atomic_save(
                output / "best.pt",
                {
                    "model_config": model_cfg,
                    "model_state": model.state_dict(),
                    "step": step,
                },
            )
        state = {
            "identity": identity,
            "model_config": model_cfg,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "batch_order": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if device.startswith("cuda") else []
            ),
            "step": step,
            "history": history,
            "best_psnr": best,
            "best_model_state": best_weights,
            "best_step": best_step,
        }
        atomic_save(last, state)
        write_json(output / "history.json", history)
        print(
            "update",
            step,
            "validation PSNR",
            round(metrics["psnr_y_db"], 4),
            flush=True,
        )
    return output
