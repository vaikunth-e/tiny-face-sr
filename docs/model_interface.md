# Model and measurement contracts

## Historical residual core

| Item | Contract |
|---|---|
| Scene | Frozen 320×240 RGB source with confirmed native-scale ROI |
| Rectangle | Half-open `(x,y,width,height)`, clipped to the frame |
| PyTorch tensor | NCHW, normalized RGB `[0,1]` |
| TFLite tensor | NHWC, batch one, 64×64×3 input and 256×256×3 output |
| Output semantics | Signed normalized RGB residual |
| Reconstruction | Byte bicubic + residual ×255 × boundary fade, then rounding/clipping |
| Geometry | 64-pixel input tile, three-pixel halo, 58-pixel valid step |
| ROI fade | 16 output pixels at the outer ROI boundary |
| Exterior invariant | Raw desktop output remains equal to the chosen bicubic baseline outside ROI |
| Operations | Four convolutions, fused ReLU where exported, DepthToSpace |

Each artifact carries its own affine quantization. Output zero points differ across widths; do not reuse one model's output parameters for another. The camera baseline explicitly accepts C16 and verifies its registry hash and geometry. A successful desktop export does not authorize arbitrary graph substitution in this camera implementation.

PyTorch PixelShuffle enumerates head channels as `[color,dy,dx]`; TensorFlow DepthToSpace expects `[dy,dx,color]`. Export permutes the final convolution filters and biases before conversion. Quantization rounds the normalized input through its actual scale/zero point, clips to signed INT8, and dequantizes output through its own parameters.

The shallow residual receptive field is 7×7 LR pixels. Width increases features but not context. New model architectures need explicit direct-RGB versus residual semantics, padding, scale, input layout, normalization, context, and full-versus-tiled agreement checks.

## Detector semantics

FOMO's center and score are not a measured head box or a calibrated correctness probability. The automatic camera mode selects the highest-scoring candidate and derives a fixed head region. Ground-truth box evaluation requires human-reviewed face bounds. Detection errors, absent model files and unreviewed scenes remain failures or missing evidence, never true negatives.

The reusable detector scorer accepts a list of truth objects with `scene_id`, `human_reviewed`, and `face_boxes`, and prediction objects with the same anonymous ID, `box_semantics=face_bounds`, and `boxes=[{"xywh":[x,y,w,h],"score":0.9}]`. It reports precision/recall, false-positive frame rate on no-face scenes and recall by source face width. Center-only outputs cannot use this box-scoring contract.

## Measurement boundaries

Model-file bytes are distinct from weight tensor bytes, tensor arena, MicroPython heap, native image buffers and physical peak memory. MAC counts cover convolutions and omit interpolation, activation, reordering, blending and copies. Desktop runtimes are not device latency predictions. Separate model loading, capture, detection, interpolation, inference, composition, JPEG encoding, transfer and complete elapsed time.

Compare raw buffers for edit localization. Comparing separately compressed JPEGs may show changes outside the edited ROI due to coding. A nonzero change count is not a quality metric. A sharper-looking face may contain generated texture rather than recovered true detail.
