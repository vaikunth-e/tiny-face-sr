"""One command interface for data, desktop models, and camera execution"""

import argparse, json
from pathlib import Path


def main(argv=None):
    """Load heavy numerical and camera dependencies only for the selected operation"""
    p = argparse.ArgumentParser(prog="tinysr", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities", help="show portable local runtime capabilities")
    model = sub.add_parser("models")
    ms = model.add_subparsers(dest="action", required=True)
    ms.add_parser("audit")
    verify = ms.add_parser("verify")
    verify.add_argument("--output", required=True, type=Path)
    prune = ms.add_parser("prune")
    prune.add_argument("--source", choices=["C16", "C24", "C32"], required=True)
    prune.add_argument("--keep", type=int, required=True)
    prune.add_argument("--output", required=True, type=Path)
    conv = ms.add_parser("convert")
    conv.add_argument("--checkpoint", type=Path, required=True)
    conv.add_argument("--calibration", type=Path)
    conv.add_argument("--output", type=Path, required=True)
    camera = sub.add_parser("camera")
    cs = camera.add_subparsers(dest="action", required=True)
    for name in ("run", "capture", "resources"):
        c = cs.add_parser(name)
        c.add_argument("--port", required=True)
        c.add_argument("--output", required=True, type=Path)
        if name == "resources":
            c.add_argument("--with-c16", action="store_true")
        if name == "run":
            c.add_argument(
                "--mode", choices=["manual", "auto", "bicubic"], default="auto"
            )
            c.add_argument("--manual", nargs=4, type=int)
            c.add_argument("--head-size", type=int, default=140)
            c.add_argument(
                "--model-path", default="/tinysr_rt1062/c16/residual_int8.tflite"
            )
    scenes = sub.add_parser("scenes")
    scenes.add_argument("--table", required=True, type=Path)
    scenes.add_argument("--output", required=True, type=Path)
    data = sub.add_parser("data")
    ds = data.add_subparsers(dest="action", required=True)
    for name in ("audit", "prepare"):
        d = ds.add_parser(name)
        d.add_argument("--manifest", type=Path, required=True)
        if name == "prepare":
            d.add_argument("--source", type=Path, required=True)
            d.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--scenes", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--backend", choices=["fp32", "int8"], default="int8")
    compare.add_argument("--device", default="auto")
    for name in ("review", "export"):
        d = sub.add_parser(name)
        d.add_argument("--results", required=True, type=Path)
        if name == "export":
            d.add_argument("--output", required=True, type=Path)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--data-root", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--device", default="cuda")
    train.add_argument("--stop-at", type=int)
    detection = sub.add_parser("score-detections")
    detection.add_argument("--truth", required=True, type=Path)
    detection.add_argument("--predictions", required=True, type=Path)
    detection.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    if a.command == "capabilities":
        from tinysr.common.device import environment

        print(json.dumps(environment(), indent=2))
    elif a.command == "models":
        if a.action == "audit":
            from tinysr.models.registry import audit

            print(json.dumps(audit(), indent=2))
        elif a.action == "verify":
            from tinysr.conversion.verify import verify

            verify(a.output)
        elif a.action == "convert":
            from tinysr.conversion.export import export

            export(a.checkpoint, a.output, a.calibration)
        else:
            import torch
            from tinysr.models.registry import model_records, selected_artifact
            from tinysr.models.runtime import TorchResidual
            from tinysr.conversion.pruning import prune_model
            from tinysr.training.engine import atomic_save

            if a.output.exists():
                raise FileExistsError("Preserve prior pruning results")
            record = next(r for r in model_records() if r["id"] == a.source)
            source = TorchResidual(
                selected_artifact(record, "best.pt"), record["channels"]
            ).model
            student, indices = prune_model(source, a.keep)
            atomic_save(
                a.output,
                {
                    "model_config": {
                        "variant": "residual",
                        "channels": a.keep,
                        "blocks": 2,
                        "scale": 4,
                    },
                    "model_state": student.state_dict(),
                    "step": 0,
                    "kept_channels": indices,
                    "quality_validated": False,
                },
            )
    elif a.command == "camera":
        if a.action == "capture":
            from tinysr.camera.capture import capture

            capture(a.port, a.output)
        elif a.action == "resources":
            from tinysr.camera.resources import probe

            probe(a.port, a.output, a.with_c16)
        else:
            from tinysr.camera.program import board_program
            from tinysr.camera.session import run_camera
            from tinysr.camera.results import save_results

            if a.output.exists():
                raise FileExistsError("Use a new camera run directory")
            script = board_program(
                mode=a.mode,
                manual=a.manual,
                head_size=a.head_size,
                model_path=a.model_path,
            )
            result, images = run_camera(a.port, script)
            save_results(a.output, result, images, script)
            if result["host_state"] != "completed":
                return 2
    elif a.command == "scenes":
        from tinysr.data.scenes import freeze

        freeze(a.table, a.output)
    elif a.command == "data":
        from tinysr.data.pairs import validate_manifest, prepare

        if a.action == "audit":
            print(json.dumps(validate_manifest(a.manifest)[1], indent=2))
        else:
            prepare(a.manifest, a.source, a.output)
    elif a.command == "compare":
        from tinysr.evaluation.compare import compare

        compare(a.scenes, a.output, a.backend, device=a.device)
    elif a.command == "review":
        from tinysr.evaluation.review import summarize_review

        print(json.dumps(summarize_review(a.results), indent=2))
    elif a.command == "export":
        from tinysr.evaluation.export import export_metrics

        export_metrics(a.results, a.output)
    elif a.command == "score-detections":
        from tinysr.evaluation.detection import score
        from tinysr.common.io import write_json

        if a.output.exists():
            raise FileExistsError("Preserve prior scoring")
        write_json(
            a.output,
            score(
                json.loads(a.truth.read_text()), json.loads(a.predictions.read_text())
            ),
        )
    else:
        from tinysr.training.engine import train

        train(a.config, a.manifest, a.data_root, a.output, a.device, a.stop_at)
    return 0
