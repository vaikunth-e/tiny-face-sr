"""Native-scale ROI reconstruction and historical image metrics"""

import time
import numpy as np
from PIL import Image
from tinysr.common.geometry import clamp, plan, verify_plan


def cost(channels, tiles=1):
    """Count dense convolution operations; one multiply-accumulate is one MAC."""
    c = int(channels)
    return {
        "parameters": 18 * c * c + 78 * c + 48,
        "convolution_macs_per_tile": 4096 * (18 * c * c + 75 * c),
        "convolution_macs_total": int(tiles) * 4096 * (18 * c * c + 75 * c),
        "excluded": ["bicubic", "bias_add", "relu", "pixel_shuffle", "blend", "copies"],
        "arena_bytes": None,
        "peak_device_ram_bytes": None,
        "device_latency_ms": None,
    }


def compose(source, baseline, roi, predict):
    """Apply the historical native-scale 64/58 tiling and 16-HR-pixel ROI fade."""
    if tuple(roi) != clamp(roi):
        raise ValueError("ROI must fit the source frame exactly")
    if source.size != (320, 240) or baseline.size != (1280, 960):
        raise ValueError("Expected QVGA source and 4x baseline")
    tiles = plan(roi, max_tiles=1000)
    check = verify_plan(roi, tiles)
    if not tiles or check["missing"] or check["overlap"]:
        raise ValueError("Invalid ROI tile coverage")
    base = np.asarray(baseline, dtype=np.uint8).copy()
    output = base.copy()
    x, y, w, h = roi
    xa, ya, xb, yb = x * 4, y * 4, (x + w) * 4, (y + h) * 4
    inference_ms = []
    for sx, sy, vx, vy, vw, vh in tiles:
        crop = source.crop((sx, sy, sx + 64, sy + 64))
        start = time.perf_counter()
        residual = predict(crop)
        inference_ms.append((time.perf_counter() - start) * 1000)
        if residual.shape != (256, 256, 3) or not np.isfinite(residual).all():
            raise ValueError("Invalid residual output")
        px, py = (vx - sx) * 4, (vy - sy) * 4
        xx, yy = vx * 4, vy * 4
        ww, hh = vw * 4, vh * 4
        gy, gx = np.ogrid[yy : yy + hh, xx : xx + ww]
        distance = np.minimum(
            np.minimum(gx - xa, xb - 1 - gx), np.minimum(gy - ya, yb - 1 - gy)
        )
        alpha = np.clip(distance / 16, 0, 1)[..., None].astype(np.float32)
        corrected = base[yy : yy + hh, xx : xx + ww].astype(np.float32)
        corrected += residual[py : py + hh, px : px + ww] * 255 * alpha
        output[yy : yy + hh, xx : xx + ww] = np.clip(
            np.floor(corrected + 0.5), 0, 255
        ).astype(np.uint8)
    difference = np.abs(output.astype(np.int16) - base.astype(np.int16))
    outside = difference.copy()
    outside[ya:yb, xa:xb] = 0
    if np.any(outside):
        raise AssertionError("Composition changed pixels outside ROI")
    return Image.fromarray(output), {
        "tiles": len(tiles),
        "changed_channels": int(np.count_nonzero(difference)),
        "sum_abs_channel_delta": int(difference.sum()),
        "outside_roi_unchanged": True,
        "desktop_tile_inference_ms": inference_ms,
    }


def paired_metrics(reference, estimate):
    """Match archived full-image gamma-coded BT.601 luma metrics on uint8 RGB."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a, b = (
        np.asarray(im, dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
        for im in (reference, estimate)
    )
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
    return {
        "psnr_y_db": psnr if np.isfinite(psnr) else None,
        "perfect_match": bool(np.array_equal(a, b)),
        "ssim_y": float(structural_similarity(a, b, data_range=255)),
    }
