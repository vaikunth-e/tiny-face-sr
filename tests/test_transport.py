"""Fragmented and tampered camera streams fail closed"""

import base64, hashlib
import pytest
from tinysr.camera.transport import Collector


def packet(data=b"{}"):
    return (
        "TINYSR|START|REPORT|%d|1|%s\nTINYSR|DATA|REPORT|0|%s\nTINYSR|END|REPORT|1\nTINYSR|DONE\n"
        % (len(data), hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode())
    )


def test_fragmented_report():
    c = Collector()
    source = packet()
    for i in range(0, len(source), 3):
        c.feed(source[i : i + 3])
    assert c.completed and c.artifact("REPORT") == b"{}"


def test_corruption_rejected():
    c = Collector()
    with pytest.raises(ValueError, match="SHA256"):
        c.feed(packet().replace("e30=", "e10="))


def test_early_done_rejected():
    with pytest.raises(ValueError):
        Collector().feed("TINYSR|DONE\n")


def test_duplicate_start_rejected():
    c = Collector()
    line = packet().splitlines()[0] + "\n"
    c.feed(line)
    with pytest.raises(ValueError):
        c.feed(line)
