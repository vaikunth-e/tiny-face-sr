import gc, image, sensor, os, time, ujson, uhashlib, ubinascii

# report resource state without assuming the gc heap represents all SDRAM
report = {"experiment": "camera_resources_v1", "stages": [], "errors": []}


def memory_snapshot():
    data = {}
    for name in ("mem_free", "mem_alloc"):
        try:
            data[name] = int(getattr(gc, name)())
        except Exception as exc:
            data[name + "_error"] = repr(exc)
    return data


def record(name, **details):
    # send a short stage marker before any potentially fragile allocation
    item = {"name": name, "memory": memory_snapshot()}
    item.update(details)
    report["stages"].append(item)
    print("TINYSR|PHASE|" + name)
    time.sleep_ms(100)
    return item


def attempt_image(name, width, height, source=None, draw=False):
    # allocation failures remain a measurement, not proof of total ram size
    gc.collect()
    start = time.ticks_ms()
    size = width * height * 2
    item = record(
        name + "_begin", width=width, height=height, theoretical_rgb565_bytes=size
    )
    buf = None
    try:
        buf = image.Image(width, height, image.RGB565)
        item["allocation_success"] = True
        item["allocation_ms"] = time.ticks_diff(time.ticks_ms(), start)
        item["actual_dimensions"] = [int(buf.width()), int(buf.height())]
        item["memory_after_allocate"] = memory_snapshot()
        if draw and source is not None:
            t = time.ticks_ms()
            buf.draw_image(source, 0, 0, x_scale=4, y_scale=4, hint=image.BICUBIC)
            item["bicubic_draw_ms"] = time.ticks_diff(time.ticks_ms(), t)
        item["operation_status"] = "completed"
    except Exception as exc:
        item["allocation_success"] = buf is not None
        item["error"] = repr(exc)
        item["operation_status"] = "failed"
    finally:
        buf = None
        gc.collect()
        item["memory_after_release"] = memory_snapshot()
    print("TINYSR|PHASE|" + name + "_end")
    time.sleep_ms(100)


try:
    record("startup")
    gc.collect()
    try:
        fs = os.statvfs("/")
        report["filesystem"] = {
            "block_size": int(fs[0]),
            "total_blocks": int(fs[2]),
            "available_blocks": int(fs[4]),
            "available_bytes_approx": int(fs[0] * fs[4]),
        }
    except Exception as exc:
        report["filesystem_error"] = repr(exc)

    # retain one real qvga capture while testing the scratch-image lifetime
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(time=1000)
    frame = sensor.snapshot()
    record("qvga_captured", source_dimensions=[int(frame.width()), int(frame.height())])

    # this strip geometry was already used by the successful v4 experiment
    attempt_image("rgb565_strip_1280x16", 1280, 16, frame, True)

    # check a neural-sized image buffer as an additional allocator reference
    attempt_image("rgb565_patch_256x256", 256, 256)

    # isolate the full-frame allocation without doing expensive full-frame drawing
    if __TEST_FULL__:
        attempt_image("rgb565_full_1280x960", 1280, 960)
    else:
        record("rgb565_full_skipped")

    # loading the known c16 model is optional and must not change model files
    if __TEST_C16__:
        try:
            files = [
                x
                for x in os.listdir("/tinysr_rt1062/c16")
                if x.lower().endswith(".tflite")
            ]
            if len(files) != 1:
                raise RuntimeError("expected exactly one installed c16 tflite model")
            import ml

            gc.collect()
            t = time.ticks_ms()
            model = ml.Model("/tinysr_rt1062/c16/" + files[0])
            record(
                "c16_loaded",
                model_bytes=int(model.len),
                reported_arena_bytes=int(model.ram),
                load_ms=time.ticks_diff(time.ticks_ms(), t),
            )
            if __TEST_FULL__:
                attempt_image("rgb565_full_with_c16_loaded", 1280, 960)
            model = None
            gc.collect()
            record("c16_released")
        except Exception as exc:
            report["errors"].append({"stage": "c16_model", "error": repr(exc)})
            record("c16_model_error")
    report["state"] = "completed"
except Exception as exc:
    import sys

    report["state"] = "failed"
    report["errors"].append({"stage": "top_level", "error": repr(exc)})
    sys.print_exception(exc)


stream("REPORT", ujson.dumps(report).encode(), 64, 60)
print(TAG + "|DONE")
while True:
    time.sleep_ms(500)
