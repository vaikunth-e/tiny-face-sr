"""Image units, coverage, checkpoint contracts and pruning regressions"""

import json, tempfile, unittest
from pathlib import Path
import numpy as np
from PIL import Image
from tinysr.models.runtime import TorchResidual
from tinysr.models.registry import model_records, selected_artifact
from tinysr.common.io import sha256, write_json
from tinysr.evaluation.compose import compose, cost
from tinysr.common.geometry import plan, verify_plan


class GeometryAndUnits(unittest.TestCase):
    def test_detector_errors_not_counted_as_negatives(self):
        from tinysr.evaluation.detection import score

        truth = [
            {
                "scene_id": "s1",
                "human_reviewed": True,
                "face_boxes": [[10, 10, 50, 50]],
            },
            {"scene_id": "s2", "human_reviewed": True, "face_boxes": []},
            {"scene_id": "s3", "human_reviewed": True, "face_boxes": []},
        ]
        predictions = [
            {
                "scene_id": "s1",
                "box_semantics": "face_bounds",
                "boxes": [{"xywh": [10, 10, 50, 50], "score": 0.9}],
            },
            {"scene_id": "s2", "box_semantics": "face_bounds", "boxes": []},
            {"scene_id": "s3", "error": "model load failed"},
        ]
        result = score(truth, predictions)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["no_face_frames_evaluated"], 1)
        self.assertEqual(result["failed_or_unreviewed_scenes"], ["s3"])
        self.assertEqual(result["status"], "incomplete")

    def test_coverage_at_edges(self):
        for roi in (
            (0, 0, 320, 240),
            (96, 56, 128, 128),
            (319, 239, 1, 1),
            (0, 120, 70, 120),
        ):
            result = verify_plan(roi, plan(roi, 1000))
            self.assertEqual(result["missing"], 0)
            self.assertEqual(result["overlap"], 0)

    def test_realistic_residual_units_and_fade(self):
        source = Image.new("RGB", (320, 240), (100, 100, 100))
        base = source.resize((1280, 960))
        out, stats = compose(
            source,
            base,
            (96, 56, 128, 128),
            lambda _: np.full((256, 256, 3), 0.04, np.float32),
        )
        self.assertEqual(out.getpixel((448, 288)), (110, 110, 110))
        self.assertEqual(out.getpixel((384, 224)), (100, 100, 100))
        self.assertTrue(stats["outside_roi_unchanged"])
        self.assertEqual(stats["tiles"], 9)

    def test_nonfinite_prediction_rejected(self):
        source = Image.new("RGB", (320, 240))
        with self.assertRaises(ValueError):
            compose(
                source,
                source.resize((1280, 960)),
                (1, 1, 1, 1),
                lambda _: np.full((256, 256, 3), np.nan),
            )

    def test_cost_matches_historical_values(self):
        for c, macs, params in (
            (16, 23789568, 5904),
            (24, 49840128, 12288),
            (32, 85327872, 20976),
        ):
            self.assertEqual(cost(c)["parameters"], params)
            self.assertEqual(cost(c)["convolution_macs_per_tile"], macs)
            self.assertIsNone(cost(c)["arena_bytes"])


class CheckpointRegression(unittest.TestCase):
    def test_native_tiling_matches_full_convolution(self):
        import torch

        row = model_records()[0]
        model = TorchResidual(selected_artifact(row, "best.pt"), 16)
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        with torch.inference_mode():
            x = torch.from_numpy(
                source.astype(np.float32).transpose(2, 0, 1).copy() / 255
            )[None]
            full = model.model.network(x)[0].permute(1, 2, 0).numpy()
        for sx, sy, vx, vy, vw, vh in plan((0, 0, 320, 240), 1000):
            patch = model.predict(source[sy : sy + 64, sx : sx + 64])
            actual = patch[
                (vy - sy) * 4 : (vy - sy + vh) * 4, (vx - sx) * 4 : (vx - sx + vw) * 4
            ]
            np.testing.assert_allclose(
                actual,
                full[vy * 4 : (vy + vh) * 4, vx * 4 : (vx + vw) * 4],
                atol=1e-6,
                rtol=1e-5,
            )

    def test_pruning_matches_masked_dense_model(self):
        import torch
        from tinysr.conversion.pruning import prune_model
        from tinysr.models.residual import TinySR

        torch.manual_seed(12)
        model = TinySR(variant="residual", channels=16)
        # use a nonzero head so this cannot pass merely because residual initialization is zero
        torch.nn.init.normal_(model.network[6].weight, std=0.02)
        student, indices = prune_model(model, 8)
        x = torch.rand(1, 3, 8, 8)
        with torch.inference_mode():
            masked = x
            for layer, kept in zip((0, 2, 4), indices):
                masked = torch.relu(model.network[layer](masked))
                mask = torch.zeros(16)
                mask[kept] = 1
                masked = masked * mask[None, :, None, None]
            expected = model.network[7](model.network[6](masked))
            actual = student.network(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertLess(
            sum(p.numel() for p in student.parameters()),
            sum(p.numel() for p in model.parameters()),
        )

    def test_actual_historical_model_state_schema(self):
        for row in model_records():
            model = TorchResidual(selected_artifact(row, "best.pt"), row["channels"])
            self.assertEqual(
                model.predict(np.zeros((64, 64, 3), np.uint8)).shape, (256, 256, 3)
            )

    def test_conflicting_architecture_is_rejected(self):
        import torch

        row = model_records()[0]
        pack = torch.load(
            selected_artifact(row, "best.pt"), weights_only=True, map_location="cpu"
        )
        pack["model_config"] = {
            "variant": "direct",
            "channels": 16,
            "blocks": 2,
            "scale": 4,
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "conflict.pt"
            torch.save(pack, path)
            with self.assertRaises(ValueError):
                TorchResidual(path, 16)
