"""Acquire a local QVGA photograph using the existing USBDBG framebuffer path"""

import time
from pathlib import Path
from .usbdbg import OpenMV


def capture(port, output):
    """Save an unmodified JPEG without fixed local ports or timestamp names"""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new capture destination")
    script = """import sensor,time
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1000)
print('TINYSR_CAPTURE_READY')
while True:
    sensor.snapshot()
    time.sleep_ms(100)
"""
    with OpenMV(port) as camera:
        camera.start(script)
        camera.wait_token("TINYSR_CAPTURE_READY")
        end = time.monotonic() + 10
        blob = None
        while blob is None and time.monotonic() < end:
            blob = camera.get_jpeg(expected=(320, 240))
            if blob is None:
                time.sleep(0.1)
        if blob is None:
            raise TimeoutError("No QVGA JPEG received")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(blob)
    return output
