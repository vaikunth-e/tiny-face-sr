def face_roi_from_center(cx, cy):
    # fOMO provides centers only so this is a provisional head-size heuristic
    width = height = HEAD_SIZE
    x = max(0, min(W - width, int(cx) - width // 2))
    y = max(0, min(H - height, int(cy) - height // 2))
    return (x, y, width, height)


def locate_face(frame):
    # use the same already-captured RGB565 frame as subsequent SR
    detector = ml.Model("fomo_face_detection")

    def postprocess(m, inputs, outputs):
        n, oh, ow, oc = m.output_shape[0]
        nms = NMS(ow, oh, inputs[0].roi)
        thresholds = [(int(math.ceil(FOMO_THRESHOLD * 255)), 255)]
        for class_index in range(oc):
            heatmap = image.Image(outputs[0][0, :, :, class_index] * 255)
            blobs = heatmap.find_blobs(
                thresholds, x_stride=1, area_threshold=1, pixels_threshold=1
            )
            for blob in blobs:
                x, y, w, h = blob.rect()
                score = (
                    heatmap.get_statistics(
                        thresholds=thresholds, roi=(x, y, w, h)
                    ).l_mean()
                    / 255.0
                )
                nms.add_bounding_box(x, y, x + w, y + h, score, class_index)
        return nms.get_bounding_boxes()

    started = millis()
    detections = detector.predict([frame], callback=postprocess)
    candidates = []
    for class_index, group in enumerate(detections):
        if class_index == 0:
            continue
        for rect, score in group:
            x, y, w, h = [int(v) for v in rect]
            cx = max(0, min(W - 1, int(math.floor(x + w / 2.0 + 0.5))))
            cy = max(0, min(H - 1, int(math.floor(y + h / 2.0 + 0.5))))
            candidates.append((cx, cy, round(float(score), 5)))
    candidates.sort(key=lambda candidate: candidate[2], reverse=True)
    selected = candidates[0] if candidates else None
    elapsed = difference(started)
    detector = None
    detections = None
    gc.collect()
    details = {
        "name": "fomo_face_detection",
        "threshold": FOMO_THRESHOLD,
        "candidate_count": len(candidates),
        "selected_center": list(selected[:2]) if selected else None,
        "selected_confidence": selected[2] if selected else None,
        "elapsed_ms": elapsed,
        "head_roi_is_heuristic": True,
        "head_side_px": HEAD_SIZE,
    }
    return (
        face_roi_from_center(selected[0], selected[1]) if selected else None
    ), details
