"""Validated USBDBG host transport reused from Camera Lab v2.

The device executes temporary MicroPython without a camera flash write.
"""

from __future__ import annotations
import io
import struct
import time
from PIL import Image

try:
    import serial
except ImportError:
    serial = None
USBDBG = 0x30
CMD = {
    "version": 0x80,
    "frame_size": 0x81,
    "frame_dump": 0x82,
    "exec": 0x05,
    "stop": 0x06,
    "running": 0x87,
    "fb_enable": 0x0D,
    "text_len": 0x8E,
    "text": 0x8F,
}


class OpenMV:
    """Single-port blocking USBDBG client; callers must serialize operations."""

    def __init__(self, port_name):
        if serial is None:
            raise RuntimeError(
                "Install pyserial: python -m pip install pyserial pillow"
            )
        self.port = serial.Serial(
            port_name, baudrate=921600, timeout=0.25, write_timeout=3
        )
        self.port_name = port_name

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.stop()
        finally:
            self.port.close()

    def cmd(self, opcode, count=0, payload=b""):
        self.port.write(struct.pack("<BBI", USBDBG, opcode, count) + payload)
        self.port.flush()

    def read(self, n, timeout=4):
        out = bytearray()
        deadline = time.monotonic() + timeout
        while len(out) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"USB read incomplete ({len(out)}/{n} bytes)")
            chunk = self.port.read(n - len(out))
            if chunk:
                out.extend(chunk)
        return bytes(out)

    def stop(self):
        self.cmd(CMD["stop"])
        time.sleep(0.13)

    def enable_fb(self):
        self.cmd(CMD["fb_enable"], 4, struct.pack("<I", 1))

    def execute(self, code):
        raw = code.encode("utf-8")
        self.cmd(CMD["exec"], len(raw), raw)

    def text(self):
        self.cmd(CMD["text_len"], 4)
        size = struct.unpack("<I", self.read(4))[0]
        if size > 200_000:
            raise RuntimeError(f"Abnormal camera text length {size}")
        if not size:
            return ""
        # avoid asking a live 4.5.9 ring buffer for its entire advertised
        # backlog: the producer can advance while that large read is pending
        count = min(size, 128)
        self.cmd(CMD["text"], count)
        return self.read(count, 5).decode("utf-8", "replace")

    def version(self):
        self.cmd(CMD["version"], 12)
        return ".".join(map(str, struct.unpack("<III", self.read(12))))

    def start(self, code):
        self.stop()
        self.text()  # clear stale console output before a new experiment
        self.enable_fb()
        self.execute(code)

    def get_jpeg(self, expected=None):
        self.cmd(CMD["frame_size"], 12)
        w, h, size = struct.unpack("<III", self.read(12))
        if not w or not h or not size:
            return None
        if expected and (w, h) != expected:
            raise RuntimeError(f"Unexpected frame {w}x{h}; expected {expected}")
        if size < 4 or size > 12_000_000:
            raise RuntimeError(f"Invalid framebuffer size {size}")
        self.cmd(CMD["frame_dump"], size)
        buf = self.read(size, timeout=max(5, size / 100_000))
        if not (buf.startswith(b"\xff\xd8") and buf.endswith(b"\xff\xd9")):
            raise RuntimeError("JPEG transfer missing valid start/end markers")
        with Image.open(io.BytesIO(buf)) as im:
            im.verify()
        return buf

    def wait_token(self, token, timeout=20):
        text = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text += self.text()
            if token in text:
                return text
            if "Traceback (most recent call last)" in text:
                raise RuntimeError("Camera exception:\n" + text)
            time.sleep(0.07)
        raise TimeoutError("No camera signal " + token + "; output:\n" + text[-3000:])
