"""Check automatic camera-local face selection and full-frame output"""

from __future__ import annotations

import base64
import binascii
import gc
import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path

# allow direct pytest invocation from the extracted project directory


import pytest
from PIL import Image

from tinysr.camera.program import board_program
from tinysr.camera.transport import Collector
from tinysr.camera.results import save_results
from tinysr.camera.session import inspect_jpeg


def decoded_console(text: str):
    """Decode the actual temporary camera-program transfer protocol"""
    receiver = Collector()
    for idx in range(0, len(text), 19):
        receiver.feed(text[idx : idx + 19])
    assert receiver.completed
    report = json.loads(receiver.artifact("REPORT"))
    return receiver, report


def fake_jpeg(rgb=(90, 110, 130)):
    """Generate actual JPEG bytes for simulated camera output"""
    output = io.BytesIO()
    Image.new("RGB", (1280, 960), rgb).save(output, "JPEG", quality=85)
    return output.getvalue()


def simulate_camera(monkeypatch, *, detected: bool):
    """Provide MicroPython-equivalent sensor, FOMO and C16 test doubles"""
    state = {
        "snapshot": None,
        "fomo_calls": 0,
        "c16_calls": 0,
        "canvas": None,
        "predicted_after_detection": True,
        "model_names": [],
    }
    base = fake_jpeg()
    enhanced = fake_jpeg((94, 115, 133))

    class JPEG(bytearray):
        def size(self):
            return len(self)

        def format(self):
            return 3

    class Frame:
        def width(self):
            return 320

        def height(self):
            return 240

        def copy(self, roi=None):
            assert roi and roi[2:] == (64, 64)
            return self

        def to_ndarray(self, code):
            assert code == "B"
            return bytes([124] * (64 * 64 * 3))

    class Canvas:
        def __init__(self, w, h, fmt):
            assert (w, h, fmt) == (1280, 960, 1)
            state["canvas"] = self
            self.colors = {}

        def size(self):
            return 2 * 1280 * 960

        def draw_image(self, frame, x, y, **kw):
            assert frame is state["snapshot"]
            assert (x, y) == (0, 0)

        def get_pixel(self, x, y):
            return self.colors.get((x, y), (120, 120, 120))

        def set_pixel(self, x, y, colors):
            self.colors[x, y] = tuple(colors)

        def to_jpeg(self, **kw):
            return JPEG(enhanced if state["c16_calls"] else base)

    class Blob:
        def rect(self):
            return (160, 120, 20, 20)

    class Stats:
        def l_mean(self):
            return 190

    class Heatmap:
        def __mul__(self, value):
            assert value == 255
            return self

    class Mask:
        def __init__(self, source):
            assert isinstance(source, Heatmap)

        def find_blobs(self, *args, **kwargs):
            return [Blob()] if detected else []

        def get_statistics(self, *args, **kwargs):
            return Stats()

    class FakeTensor:
        def __getitem__(self, index):
            assert isinstance(index, tuple)
            return Heatmap()

    class NMS:
        def __init__(self, w, h, roi):
            assert len(roi) == 4
            self.faces = []

        def add_bounding_box(self, x0, y0, x1, y1, score, class_index):
            if class_index == 1:
                self.faces.append(((x0, y0, x1 - x0, y1 - y0), score))

        def get_bounding_boxes(self):
            return [[], self.faces]

    class FomoModel:
        output_shape = ((1, 10, 10, 2),)

        def predict(self, inputs, callback):
            assert inputs == [state["snapshot"]]
            state["fomo_calls"] += 1
            inp = types.SimpleNamespace(roi=(0, 0, 320, 240))
            return callback(self, [inp], [FakeTensor()])

    class C16Model:
        len = 11152
        ram = 394992
        input_shape = ((1, 64, 64, 3),)
        output_shape = ((1, 256, 256, 3),)
        input_zero_point = (-128,)
        input_scale = (1 / 255,)

        def predict(self, inputs, callback):
            assert state["fomo_calls"] in (0, 1)
            state["c16_calls"] += 1
            tensor = types.SimpleNamespace(
                reshape=lambda shape: [0.04] * (256 * 256 * 3)
            )
            return callback(self, inputs, [tensor])

    def model_factory(path):
        state["model_names"].append(path)
        if path == "fomo_face_detection":
            return FomoModel()
        if str(path).endswith(".tflite"):
            return C16Model()
        raise AssertionError("unexpected model path")

    imgmod = types.ModuleType("image")
    imgmod.RGB565 = 1
    imgmod.BICUBIC = 2
    imgmod.JPEG = 3
    imgmod.Image = lambda *args: Mask(*args) if len(args) == 1 else Canvas(*args)
    sensor = types.ModuleType("sensor")
    sensor.RGB565 = 1
    sensor.QVGA = 2
    sensor.reset = lambda: None
    sensor.set_pixformat = lambda _: None
    sensor.set_framesize = lambda _: None
    sensor.skip_frames = lambda **kw: None

    def snapshot():
        state["snapshot"] = Frame()
        return state["snapshot"]

    sensor.snapshot = snapshot

    ml = types.ModuleType("ml")
    ml.Model = model_factory
    mlutils = types.ModuleType("ml.utils")
    mlutils.NMS = NMS
    ml.utils = mlutils

    fake_time = types.ModuleType("time")
    counter = [0]

    def millis():
        counter[0] += 2
        return counter[0]

    fake_time.ticks_ms = millis
    fake_time.ticks_diff = lambda end, start: end - start

    def sleep(milliseconds):
        if milliseconds == 500:
            raise SystemExit("simulated board finished")

    fake_time.sleep_ms = sleep

    fake_json = types.ModuleType("ujson")
    fake_json.dumps = json.dumps
    fake_hash = types.ModuleType("uhashlib")
    fake_hash.sha256 = hashlib.sha256
    fake_binary = types.ModuleType("ubinascii")
    fake_binary.hexlify = binascii.hexlify
    fake_binary.b2a_base64 = base64.b64encode
    fake_gc = types.ModuleType("gc")
    fake_gc.collect = lambda: None
    fake_gc.mem_free = lambda: 5_000_000
    fake_gc.mem_alloc = lambda: 3_000_000

    for name, mod in {
        "sensor": sensor,
        "image": imgmod,
        "ml": ml,
        "ml.utils": mlutils,
        "time": fake_time,
        "ujson": fake_json,
        "uhashlib": fake_hash,
        "ubinascii": fake_binary,
        "gc": fake_gc,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(gc, "collect", lambda: None)
    monkeypatch.setattr(os, "listdir", lambda path: ["c16_weights.tflite"])
    import builtins

    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, mode="r", *args, **kwargs: (
            io.BytesIO(b"local-test-weights")
            if str(path).endswith(".tflite")
            else real_open(path, mode, *args, **kwargs)
        ),
    )

    return state, base, enhanced


def test_automatic_face_selection_and_full_c16_output(monkeypatch, capsys):
    """Use the same frozen camera frame for FOMO, bicubic and C16"""
    state, base, enhanced = simulate_camera(monkeypatch, detected=True)
    script = board_program(manual=None, head_size=32)
    from tinysr.models.registry import model_records

    original = model_records()[0]["artifacts"]["residual_int8.tflite"]["sha256"]
    script = script.replace(original, hashlib.sha256(b"local-test-weights").hexdigest())

    with pytest.raises(SystemExit, match="simulated board finished"):
        exec(compile(script, "<camera-test>", "exec"), {})
    receiver, report = decoded_console(capsys.readouterr().out)
    assert report["state"] == "completed"
    assert report["roi_mode"] == "auto"
    assert report["detector"]["candidate_count"] == 1
    assert report["detector"]["selected_center"] == [170, 130]
    assert report["roi"] == [154, 114, 32, 32]
    assert report["tiles"] == 1
    assert report["neural_applied"] is True
    assert state["fomo_calls"] == 1
    assert state["c16_calls"] == 1
    assert report["neural"]["changed_pixels"] > 0
    assert receiver.artifact("BASE") == base
    assert receiver.artifact("NEURAL") == enhanced


def test_no_face_returns_explicit_bicubic_only_output(monkeypatch, capsys):
    """Avoid running C16 on arbitrary regions when FOMO finds no face"""
    state, base, _ = simulate_camera(monkeypatch, detected=False)
    script = board_program(manual=None, head_size=32)
    from tinysr.models.registry import model_records

    original = model_records()[0]["artifacts"]["residual_int8.tflite"]["sha256"]
    script = script.replace(original, hashlib.sha256(b"local-test-weights").hexdigest())
    with pytest.raises(SystemExit, match="simulated board finished"):
        exec(compile(script, "<camera-no-face-test>", "exec"), {})
    receiver, report = decoded_console(capsys.readouterr().out)
    assert report["state"] == "completed"
    assert report["detector"]["candidate_count"] == 0
    assert report["neural_applied"] is False
    assert report["roi"] is None
    assert report["tiles"] == 0
    assert report["neural"]["changed_pixels"] == 0
    assert state["fomo_calls"] == 1
    assert state["c16_calls"] == 0
    assert receiver.artifact("BASE") == receiver.artifact("NEURAL") == base


@pytest.mark.parametrize(
    "config",
    [
        {"fomo_threshold": 0.01},
        {"head_size": 31},
        {"head_size": 181},
        {"manual": (96, 56, 0, 128)},
    ],
)
def test_invalid_auto_settings_are_rejected(config):
    """Prevent ambiguous face-ROI or detector configuration"""
    with pytest.raises(ValueError):
        board_program(**config)


def test_bicubic_mode_never_loads_neural_models(monkeypatch, capsys):
    state, base, _ = simulate_camera(monkeypatch, detected=True)
    script = board_program(mode="bicubic")
    with pytest.raises(SystemExit):
        exec(compile(script, "camera-bicubic-test", "exec"), {})
    receiver, report = decoded_console(capsys.readouterr().out)
    assert report["roi_mode"] == "bicubic"
    assert state["model_names"] == []
    assert report["tiles"] == 0 and not report["neural_applied"]
    assert receiver.artifact("BASE") == receiver.artifact("NEURAL") == base


def test_manual_mode_bypasses_detector(monkeypatch, capsys):
    state, _, _ = simulate_camera(monkeypatch, detected=False)
    script = board_program(manual=(100, 60, 32, 32))
    from tinysr.models.registry import model_records

    original = model_records()[0]["artifacts"]["residual_int8.tflite"]["sha256"]
    script = script.replace(original, hashlib.sha256(b"local-test-weights").hexdigest())
    with pytest.raises(SystemExit):
        exec(compile(script, "camera-manual-test", "exec"), {})
    _, report = decoded_console(capsys.readouterr().out)
    assert state["fomo_calls"] == 0 and state["c16_calls"] == 1
    assert report["roi"] == [100, 60, 32, 32] and report["roi_mode"] == "manual"
