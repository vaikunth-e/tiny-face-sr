"""Metadata-only framebuffer and optional model-overlap allocation probe"""

from importlib.resources import files
from pathlib import Path
import json, time
from tinysr.common.io import write_json
from .transport import Collector
from .usbdbg import OpenMV


def probe(port, output, with_c16=False):
    """Preserve local diagnostics without calling heap counters physical peak RAM"""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new resource-probe destination")
    script = (
        files("tinysr.device.openmv")
        .joinpath("resources.py")
        .read_text()
        .replace("__TEST_FULL__", "True")
        .replace("__TEST_C16__", repr(bool(with_c16)))
    )
    prefix = "import gc,time,uhashlib,ubinascii\nTAG='TINYSR'\nMAX_JPEG_BYTES=3000000\n"
    prefix += files("tinysr.device.openmv").joinpath("transport.py").read_text()
    script = prefix + "\n" + script
    compile(script, "resource_program", "exec")
    collector = Collector()
    with OpenMV(port) as camera:
        camera.start(script)
        end = time.monotonic() + 180
        while not collector.completed and time.monotonic() < end:
            collector.feed(camera.text())
            time.sleep(0.025)
    if not collector.completed:
        raise TimeoutError("Resource report did not complete")
    report = json.loads(collector.artifact("REPORT"))
    write_json(output, report)
    return output
