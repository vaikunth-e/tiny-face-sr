"""Deterministic QVGA->4x face region geometry shared by tests and host.

All source rectangles are xywh, half-open and clipped to [0,W)x[0,H).
"""

W, H, SCALE, TILE, HALO, VALID = 320, 240, 4, 64, 3, 58


def clamp(rect):
    if rect is None:
        return None
    x, y, w, h = (int(t) for t in rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def expand(face, margin=0.35):
    """Expand a Haar xywh box by a fraction on EACH side."""
    b = clamp(face)
    if not b:
        return None
    x, y, w, h = b
    import math

    ax = int(math.floor(x - margin * w + 0.5))
    ay = int(math.floor(y - margin * h + 0.5))
    bx = int(math.floor(x + w + margin * w + 0.5))
    by = int(math.floor(y + h + margin * h + 0.5))
    return clamp((ax, ay, bx - ax, by - ay))


def plan(roi, max_tiles=16):
    """Yield (input_tile_x,input_tile_y,valid_x,valid_y,valid_w,valid_h).

    Valid rectangles are disjoint and cover the selected ROI exactly once.
    Input origin clamps to full QVGA, never changes the 64x64 model input.
    """
    r = clamp(roi)
    if r is None:
        return []
    x, y, w, h = r
    tiles = []
    for vy in range(y, y + h, VALID):
        vh = min(VALID, y + h - vy)
        sy = max(0, min(vy - HALO, H - TILE))
        if vy + vh > sy + TILE:
            sy = vy + vh - TILE
        for vx in range(x, x + w, VALID):
            vw = min(VALID, x + w - vx)
            sx = max(0, min(vx - HALO, W - TILE))
            if vx + vw > sx + TILE:
                sx = vx + vw - TILE
            tiles.append((sx, sy, vx, vy, vw, vh))
    return tiles if len(tiles) <= max_tiles else []


def verify_plan(roi, tiles):
    """Return exact-coverage errors without retaining a full output image."""
    roi = clamp(roi)
    if not roi:
        return {"covered": 0, "expected": 0, "overlap": 0, "missing": 0}
    x, y, w, h = roi
    coverage = bytearray(w * h)
    overlap = 0
    for sx, sy, vx, vy, vw, vh in tiles:
        assert 0 <= sx <= W - TILE and 0 <= sy <= H - TILE
        assert vx >= sx and vy >= sy and vx + vw <= sx + TILE and vy + vh <= sy + TILE
        for py in range(vy, vy + vh):
            for px in range(vx, vx + vw):
                idx = (py - y) * w + (px - x)
                assert 0 <= idx < len(coverage)
                if coverage[idx]:
                    overlap += 1
                coverage[idx] = 1
    return {
        "covered": sum(coverage),
        "expected": w * h,
        "overlap": overlap,
        "missing": len(coverage) - sum(coverage),
    }
