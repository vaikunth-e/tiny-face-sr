def find_model():
    return MODEL_PATH


def model_sha256(path):
    # identify the installed model without transmitting any weights
    digest = uhashlib.sha256()
    with open(path, "rb") as stream_file:
        while True:
            part = stream_file.read(4096)
            if not part:
                break
            digest.update(part)
    return ubinascii.hexlify(digest.digest()).decode()


def validated_model(path):
    m = ml.Model(path)
    if int(m.len) != MODEL_BYTES:
        raise RuntimeError("unexpected model file size")
    if len(m.input_shape) != 1 or tuple(m.input_shape[0]) != (1, 64, 64, 3):
        raise RuntimeError("unexpected model input geometry")
    if len(m.output_shape) != 1 or tuple(m.output_shape[0]) != (1, 256, 256, 3):
        raise RuntimeError("unexpected model output geometry")
    if m.input_zero_point[0] != -128 or abs(m.input_scale[0] - 1 / 255.0) > 0.00001:
        raise RuntimeError("unexpected input quantization")
    return m


def preprocess(pixels):
    # openMV inference callback expects signed int8 inputs in raw byte storage
    class Fill:
        def __call__(self, dest, shape, dtype):
            if len(dest) != len(pixels):
                raise RuntimeError("C16 input byte length mismatch")
            for k in range(len(pixels)):
                dest[k] = pixels[k] ^ 128

    return Fill()


def apply_tile(frame, full, model, tile, roi):
    sx, sy, vx, vy, vw, vh = tile
    img = frame.copy(roi=(sx, sy, TILE, TILE))
    pixels = bytes(img.to_ndarray("B"))
    img = None
    if len(pixels) != 64 * 64 * 3:
        raise RuntimeError("unexpected input tile byte count")

    def get_output(m, inputs, outputs):
        return outputs[0]

    begin = millis()
    result = model.predict([preprocess(pixels)], callback=get_output)
    prediction_ms = difference(begin)
    residual = result.reshape((256 * 256 * 3,))
    if len(residual) != 256 * 256 * 3:
        raise RuntimeError("unexpected C16 residual geometry")

    # update the camera-owned complete RGB565 image in place
    # count differences in the STORED RGB565 result, not hypothetical float pixels
    bx, by, bw, bh = roi
    rx0, ry0, rx1, ry1 = bx * 4, by * 4, (bx + bw) * 4, (by + bh) * 4
    changed_pixels = 0
    changed_channels = 0
    sum_abs_delta = 0
    begin = millis()
    for ry in range(vh * 4):
        gy = vy * 4 + ry
        ty = gy - sy * 4
        for rx in range(vw * 4):
            gx = vx * 4 + rx
            tx = gx - sx * 4
            distance = min(gx - rx0, rx1 - 1 - gx, gy - ry0, ry1 - 1 - gy)
            alpha = min(1.0, max(0.0, distance / 16.0))
            if alpha <= 0.0:
                continue
            before = full.get_pixel(gx, gy)
            offset = (ty * 256 + tx) * 3
            rgb = [0, 0, 0]
            for c in range(3):
                v = int(before[c] + float(residual[offset + c]) * 255.0 * alpha + 0.5)
                rgb[c] = max(0, min(255, v))
            full.set_pixel(gx, gy, (rgb[0], rgb[1], rgb[2]))
            after = full.get_pixel(gx, gy)
            differences = [abs(after[c] - before[c]) for c in range(3)]
            if differences[0] or differences[1] or differences[2]:
                changed_pixels += 1
                changed_channels += (
                    int(differences[0] > 0)
                    + int(differences[1] > 0)
                    + int(differences[2] > 0)
                )
                sum_abs_delta += differences[0] + differences[1] + differences[2]
    compose_ms = difference(begin)
    result = None
    residual = None
    pixels = None
    gc.collect()
    return {
        "predict_ms": prediction_ms,
        "compose_ms": compose_ms,
        "changed_pixels": changed_pixels,
        "changed_channels": changed_channels,
        "sum_abs_delta": sum_abs_delta,
    }
