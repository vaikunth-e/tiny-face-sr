start_total = millis()
try:
    phase("startup")
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(time=1000)
    frame = sensor.snapshot()
    if (frame.width(), frame.height()) != (W, H):
        raise RuntimeError("unexpected camera capture geometry")
    phase("capture", width=W, height=H)
    if MODE == "bicubic":
        roi, detector_result = None, None
        phase("bicubic_only_selected")
    elif MANUAL is None:
        roi, detector_result = locate_face(frame)
        report["detector"] = detector_result
        phase(
            "face_detected" if roi else "no_face_candidates",
            candidates=detector_result["candidate_count"],
            milliseconds=detector_result["elapsed_ms"],
        )
    else:
        roi = clip_roi(MANUAL)
        phase("manual_roi_selected")
    tiles = tiles_for(roi) if roi else []
    if len(tiles) > MAX_TILES:
        raise RuntimeError("selected region exceeds neural tile budget")
    report["roi"] = roi
    report["tiles"] = len(tiles)
    report["neural_applied"] = bool(tiles)
    full = image.Image(OUT_W, OUT_H, image.RGB565)
    phase("full_buffer_allocated", frame_bytes=int(full.size()))
    begin = millis()
    full.draw_image(frame, 0, 0, x_scale=4, y_scale=4, hint=image.BICUBIC)
    phase("bicubic_drawn", milliseconds=difference(begin))

    # send the bicubic baseline from the exact same captured frame
    begin = millis()
    base = full.to_jpeg(copy=True, quality=QUALITY)
    if base.format() != image.JPEG:
        raise RuntimeError("bicubic output encoding failed")
    phase(
        "bicubic_encoded", milliseconds=difference(begin), jpeg_bytes=int(base.size())
    )
    begin = millis()
    stream("BASE", base, TX_BYTES, TX_PAUSE)
    base = None
    phase("bicubic_transferred", milliseconds=difference(begin))

    # load C16 only when a face ROI is available
    totals = {
        "inference_ms": 0,
        "composition_ms": 0,
        "changed_pixels": 0,
        "changed_channels": 0,
        "sum_abs_delta": 0,
    }
    model = None
    if tiles:
        begin = millis()
        model_path = find_model()
        model_digest = model_sha256(model_path)
        if model_digest != MODEL_SHA256:
            raise RuntimeError("C16 model identity mismatch")
        model = validated_model(model_path)
        phase(
            "model_loaded",
            milliseconds=difference(begin),
            model_bytes=int(model.len),
            arena_bytes=int(model.ram),
            model_sha256=model_digest,
        )
        for n, tile in enumerate(tiles):
            r = apply_tile(frame, full, model, tile, roi)
            totals["inference_ms"] += r["predict_ms"]
            totals["composition_ms"] += r["compose_ms"]
            for key in ("changed_pixels", "changed_channels", "sum_abs_delta"):
                totals[key] += r[key]
            print(TAG + "|PHASE|tile_%02d" % (n + 1))
            time.sleep_ms(70)
    else:
        phase("neural_skipped_no_face")
    report["neural"] = totals
    phase("neural_composed", neural_tiles=len(tiles))

    # encode a second complete image; no-face control remains bicubic
    begin = millis()
    neural = full.to_jpeg(copy=True, quality=QUALITY)
    if neural.format() != image.JPEG:
        raise RuntimeError("neural output encoding failed")
    phase(
        "neural_encoded", milliseconds=difference(begin), jpeg_bytes=int(neural.size())
    )
    begin = millis()
    stream("NEURAL", neural, TX_BYTES, TX_PAUSE)
    neural = None
    phase("neural_transferred", milliseconds=difference(begin))
    model = None
    full = None
    frame = None
    phase("objects_released")
    report["total_camera_ms_before_report_transfer"] = difference(start_total)
    report["state"] = "completed"
except Exception as exc:
    report["state"] = "failed"
    report["errors"].append(repr(exc))
    print(TAG + "|ERROR|" + repr(exc))
    sys.print_exception(exc)

try:
    blob = ujson.dumps(report).encode()
    stream("REPORT", blob, 64, 35)
    print(TAG + "|DONE")
except Exception as exc:
    print(TAG + "|FATAL|" + repr(exc))
while True:
    time.sleep_ms(500)
