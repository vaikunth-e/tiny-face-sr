"""Receive complete camera-generated JPEGs without host composition"""

import io
import json
import time
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw
from .transport import Collector
from .usbdbg import OpenMV

SIZE = (1280, 960)


def inspect_jpeg(blob: bytes) -> dict:
    """verify a finished JPEG without modifying or reconstructing its pixels"""
    if blob[:2] != b"\xff\xd8" or blob[-2:] != b"\xff\xd9":
        raise ValueError("invalid JPEG markers")
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        if image.format != "JPEG" or image.size != SIZE:
            raise ValueError("unexpected finished JPEG format or dimensions")
    return {"bytes": len(blob), "width": SIZE[0], "height": SIZE[1]}


def run_camera(
    port: str, script: str, *, timeout: int = 900, max_jpeg_bytes: int = 3_000_000
) -> tuple[dict, dict]:
    """start camera program and collect only full checksummed image artifacts"""
    collector = Collector(max_jpeg_bytes=max_jpeg_bytes)
    result = {"host_state": "started", "camera_errors": []}
    images = {}
    try:
        with OpenMV(port) as camera:
            result["firmware"] = camera.version()
            camera.start(script)
            deadline = time.monotonic() + timeout
            next_report = 0.0
            while time.monotonic() < deadline:
                text = camera.text()
                if text:
                    current = len(collector.phase_log)
                    collector.feed(text)
                    for phase in collector.phase_log[current:]:
                        print("Camera:", phase)
                    if collector.completed:
                        break
                else:
                    time.sleep(0.025)
                if time.monotonic() > next_report:
                    for kind in ("BASE", "NEURAL"):
                        transfer = collector.transfers.get(kind)
                        if transfer and not transfer["finished"]:
                            print(
                                "  %s transfer: %d/%d bytes"
                                % (kind, len(transfer["data"]), transfer["length"])
                            )
                    next_report = time.monotonic() + 25
            if not collector.completed:
                raise TimeoutError("camera transfer incomplete or timed out")
        device = json.loads(collector.artifact("REPORT"))
        result["device"] = device
        if device.get("state") != "completed":
            raise RuntimeError("device error: " + repr(device.get("errors")))
        for kind in ("BASE", "NEURAL"):
            image_data = collector.artifact(kind)
            images[kind] = image_data
            image_report = inspect_jpeg(image_data)
            stage = "bicubic_encoded" if kind == "BASE" else "neural_encoded"
            matching = [s["jpeg_bytes"] for s in device["stages"] if s["name"] == stage]
            if len(matching) != 1 or matching[0] != image_report["bytes"]:
                raise ValueError(kind + " device JPEG byte count disagrees with host")
        result["host_state"] = "completed"
        result["images"] = {kind: inspect_jpeg(blob) for kind, blob in images.items()}
    except Exception as exc:
        result["host_state"] = "failed"
        result["error"] = repr(exc)
        if (
            "REPORT" in collector.transfers
            and collector.transfers["REPORT"]["finished"]
        ):
            try:
                result["device"] = json.loads(collector.artifact("REPORT"))
            except Exception:
                pass
    result["camera_phases"] = collector.phase_log
    result["camera_errors"] = collector.errors
    result["camera_console_tail"] = "\n".join(collector.tail[-120:])
    if result["host_state"] != "completed":
        images = {}
    return result, images


def local_comparison(images: dict, folder: Path, *, neural_applied: bool) -> None:
    """create a local review visualization without modifying camera image files"""
    with Image.open(io.BytesIO(images["BASE"])) as base, Image.open(
        io.BytesIO(images["NEURAL"])
    ) as neural:
        base_rgb = base.convert("RGB")
        neural_rgb = neural.convert("RGB")
        sheet = Image.new("RGB", (SIZE[0] * 2, SIZE[1] + 38), "#202020")
        sheet.paste(base_rgb, (0, 38))
        sheet.paste(neural_rgb, (SIZE[0], 38))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), "Camera bicubic (same frame)", fill="white")
        draw.text(
            (SIZE[0] + 12, 10),
            (
                "Camera C16 neural (same frame)"
                if neural_applied
                else "No face: bicubic-only final"
            ),
            fill="white",
        )
        sheet.save(folder / "compare_camera_bicubic_vs_c16.png")
        # compressed JPEG differences reflect coding changes as well as neural edits
        difference = ImageChops.difference(base_rgb, neural_rgb)
        difference.save(folder / "jpeg_difference_visualization.png")
