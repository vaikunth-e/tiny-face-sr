# Technical findings and failure record

This record orders evidence by experiment and software version. It omits private chronology, original transcripts, local machine inventories, paths and personal attribution. Detailed private source/result archives remain external provenance. Values below are inherited measurements unless explicitly identified as migration checks. They must not be relabeled as measurements of the refactored release.

## Dataset and scoring protocol

The original aligned-face task uses FFHQ RGB sources, Lanczos reduction to 256×256 HR, bicubic reduction to 64×64 LR and bicubic enlargement as the reconstruction control. Original manifests fix 100 validation and 100 reserved test images. Training grew from 500 to 5,000 and then 20,000 nested images. The supplied public manifests preserve membership and ordering but omit private preparation settings and local path histories. Membership is image-disjoint; identity-disjointness has not been demonstrated.

Training used RGB L1 and Adam. Reconstruction metrics use gamma-coded BT.601 luma `0.299R + 0.587G + 0.114B`, range 255, SSIM window 7 and means of per-image values. Validation is not the reserved neural test split. Test metrics from the original interpolation experiment predate neural selection and must not justify later test-driven tuning.

| Classical interpolation | Validation PSNR | Validation SSIM | Original interpolation test PSNR | Test SSIM |
|---|---:|---:|---:|---:|
| Nearest | 26.2725 | 0.773471 | 26.0286 | 0.773475 |
| Bilinear | 27.5554 | 0.811549 | 27.3585 | 0.812114 |
| Bicubic | 28.2834 | 0.832368 | 28.0781 | 0.832572 |

## Dense baseline studies

The direct and bicubic-residual architectures have four convolutions: 3→C, C→C, C→C with 3×3 kernels, then C→48 with 1×1, followed by PixelShuffle ×4. Residual heads initialize at zero so initial reconstruction equals bicubic. There is no pooling, normalization, attention or intermediate resolution change. The two additional processing blocks are sequential pairs, not individual residual blocks.

| Width | Parameters | Convolution MACs per 64×64 tile | INT8 file bytes |
|---|---:|---:|---:|
| C16 | 5,904 | 23,789,568 | 11,152 |
| C24 | 12,288 | 49,840,128 | 18,184 |
| C32 | 20,976 | 85,327,872 | 27,520 |

The pilot 5,000-image C16 residual model reached validation PSNR 29.6553 and SSIM 0.864985 after 20 epochs, versus direct 28.5910/0.842618. The direct best checkpoint scored 28.6158; final and best checkpoints are not interchangeable.

The equal-update factorial study used 6,000 Adam updates and batches of 16. Architecture × data size × width produced eight cells. Bicubic validation PSNR was 28.2834.

| Architecture | Training images | Width | Validation PSNR | Validation SSIM |
|---|---:|---:|---:|---:|
| Direct | 5,000 | 16 | 28.5621 | 0.83818 |
| Direct | 20,000 | 16 | 28.5205 | 0.83828 |
| Direct | 5,000 | 32 | 28.7562 | 0.84678 |
| Direct | 20,000 | 32 | 28.7076 | 0.84501 |
| Residual | 5,000 | 16 | 29.6193 | 0.86406 |
| Residual | 20,000 | 16 | 29.6732 | 0.86521 |
| Residual | 5,000 | 32 | 29.9189 | 0.87004 |
| Residual | 20,000 | 32 | 29.9196 | 0.87008 |

Continuation to 25,000 updates used learning rates 0.001 through update 12,000, 0.0005 through 18,000 and 0.00025 afterward. Best validation PSNRs were 29.724019 (5k/C16), 29.978845 (5k/C32), 29.779038 (20k/C16), and 30.002708 (20k/C32). The 5k/C32 best occurred at update 24,000; the other best values occurred at 25,000. Additional data helped modestly at matched updates. Unequal-epoch comparisons also change exposures and optimization schedule and cannot isolate dataset size.

The separate CUDA width study reached 29.730633/0.866313 for C16, 29.893461/0.869610 for C24, and 29.982991/0.871393 for C32 at selected best checkpoints. C32's best was at update 22,000; the other best checkpoints were at 25,000. One-seed image-bootstrap comparisons found C24−C16 +0.162828 dB and C32−C24 +0.089530 dB. These intervals omit training-seed and checkpoint-selection uncertainty. The selected preserved models combine the stronger complementary C16/C32 with the CUDA C24; this cross-study selection is not a controlled estimate of width effects.

Fine curls, beard/skin texture and small ornaments remained less resolved than HR references. Similar run times across widths despite different MACs motivated profiling data/metric/host overhead; they do not prove CUDA reduced model quality or guarantee speedup. Broad width sweeps or longer training had diminishing value. Stronger architecture, application matching and perceptual objectives became the next quality questions.

## Checkpoint and conversion lessons

An earlier report loader expected top-level configuration while continuation checkpoints stored it under `frozen_identity`; another loader omitted `model_state`. Repaired loading accepts supported top-level/nested configuration, rejects conflicts and loads the exact state strictly. Do not infer architecture from a filename.

PyTorch PixelShuffle and TensorFlow DepthToSpace head ordering differ. Exact filter/bias permutation is required. Baseline conversion preserved signed normalized residual semantics and separate bicubic reconstruction. Each quantized artifact has distinct output scales/zero points. Desktop kernels can differ by several quantization levels without implying a universal tolerance or camera parity result.

Recovered checkpoints and exports were hash-verified. Reconstructed archived image controls and eight available validation examples supported conversion reproduction, but did not rerun the whole 100-image validation set. Earlier integer golden deviations of 2/3/3 output levels for C16/C24/C32 were retained as conditional kernel observations; provisional tolerance failures were not silently discarded.

Distribution checkpoints in v1.0.0 contain model configuration, tensors, step and a schema tag only. All learned tensors match recovered originals. This deliberately changes checkpoint file hashes. The registry retains original artifact hashes and new distribution hashes; TFLite bytes are unchanged. Original optimizer checkpoints/private run records are not distributed. The new trainer resumes its own format, not arbitrary historical optimizer states.

## Device numerical and allocation history

On the firmware compatibility baseline, all three widths loaded and executed. Earlier reported arenas were 394,992/395,184/395,376 bytes. A repeated-predict probe measured approximately 113.8265/183.0080/261.1985 ms medians. Those calls included input/API and legacy dequantized callback overhead and excluded capture, interpolation, composition, transfer and load; they are not pure convolution times or end-to-end output times.

The first contiguous RGB888 1280×960 buffer required 3,686,400 bytes and failed allocation. A streamed diagnostic avoided the allocation but originally assembled strips/patches on the host; it demonstrated numerical integration, not a fully camera-local finished output. A subsequent missing ×255 residual scaling error produced an apparent no-op. A one-tile diagnostic established the required normalized-to-byte scaling before the full pipeline was rerun.

The corrected streamed v4 control used nine tiles in a 128×128 source ROI and 60 bicubic strips. Camera and host changed-channel counts agreed at 319,645; sum absolute channel delta was 539,272. Nine predictions totaled 1,398 ms, but the entire paced diagnostic took 537,954 ms. The strip stage included transfer; it was not pure interpolation or pure transport. Channel changes were small and did not establish perceptual quality. Native QVGA remained visually competitive with the generated result in the development assessment.

Resource probe v6.1 later allocated a contiguous RGB565 1280×960 image of 2,457,600 bytes. It also allocated a 40,960-byte strip and a 131,072-byte patch. Full allocation took 173 ms in one initial run; a C16-overlap run loaded the 11,152-byte model and allocated the full frame in 174 ms while reporting 5,643,520 bytes free MicroPython heap. The heap observations did not measure physical peak RAM or prove repeated-run stability.

Delayed heap recovery after clearing an image reference was not proof of permanent leakage: some later stages recovered earlier allocations. Buffer lifetimes, native allocation and Python reference lifetime need explicit repeated-capture testing. One filesystem observation had 389,120 bytes free; that was a transient writable-volume state, not a fundamental model-size ceiling. Optional external storage and model placement remain separate deployment choices.

Full bicubic composition v6.2 physically generated the entire RGB565 frame in-camera: allocation 20 ms, bicubic draw 243 ms, JPEG encode 277 ms in that run. The host verified the finished image dimensions and transfer digest. No neural inference occurred in that bicubic-only experiment.

Manual neural composition v6.3 physically generated complete bicubic and enhanced JPEGs on the camera with C16 co-resident. Nine tiles required 1,394 ms inference and 52,116 ms composition. It changed 37,334 stored RGB565 pixels and 72,183 channel values, with absolute delta sum 440,515. The two JPEGs were 166,973 and 171,816 bytes. Deliberately paced transfers took 61,684 and 63,445 ms; total camera work before report transfer was 185,362 ms. The minimum sampled free heap was 4,680,640 bytes. This was single-run numerical/system feasibility, not sustained operation or improved image quality.

Automatic v6.4 combined FOMO and C16 on the same captured frame, released FOMO before C16 loading, selected a provisional 140×140 head rectangle and skipped C16 when no face was found. It passed offline positive/no-face/manual tests, but no physical combined result was established. The refactor preserves that distinction.

## Detector and transport failures

Desktop OpenCV found faces in saved images where embedded Haar did not. That controlled result established desktop detector behavior, not camera detector success. Native grayscale and parameter diagnostics did not justify repeated blind threshold sweeps. FOMO later provided on-camera positive examples with human-reviewed coverage; center localization still did not measure scale.

OpenMV's stump-format Haar representation cannot faithfully encode tree-based weak classifiers such as the tested `alt2` cascade. Truncating unsupported trees would invalidate conclusions. The checked converter rejects unsupported layouts, validates fixed-point range, geometry and feature indices, then reparses generated binaries. Desktop conversion or format readback is not an embedded load/detection result.

One custom-cascade run failed to return an intact report and recorded too little raw diagnostic evidence. A bounded, numbered, paced, checksum-framed report plus persistent local phase logs restored observability. The precise cause of the earlier lost report remained a hypothesis. Later runs exposed missing converted files, then a root-mount path issue. Missing files and invalid rectangles must not count as valid zero detections or support an algorithm comparison. A load-successful default cascade remained negative on the tested scene; an alternate produced invalid geometry. FOMO was retained as the positive control without claiming broad superiority across all scenes.

Large unpaced console records can be lost through the USBDBG ring-buffer path. Keep bounded chunks, pacing, ordering, count and SHA256 validation. Preserve a useful local failure receipt. Do not treat an intact host session, a DONE marker alone, or an allocation-only probe as completion of every downstream gate. Transfer optimization must retain correctness checks.

## Real camera quality and reporting

The single-image INT8 development comparison used one confirmed native ROI and nine tiles per model. All pixels outside the selected desktop ROI remained unchanged. Completed review reported no preference wins over bicubic and no objectionable artifacts. Because the original summary combined ties and losses, it cannot establish how many of each occurred or rank models by direct comparisons.

FP32 inference completed on the same scene; all nine review rows remained pending in its returned report. Qualitative assessment still found marginal benefit across widths. Quantization was therefore unlikely to be the sole explanation, but direct FP32/INT8 numerical equivalence or a completed blinded ranking was not established. Possible causes include shallow context, whole-face versus native-patch training mismatch, conservative reconstruction loss and camera degradation differences. They remain hypotheses.

The unified review retains wins, ties and losses, direct model pairs, artifact flags, missing review rows, and scene-weighted rates. Changed pixels, larger dimensions or successful inference cannot replace useful visual improvement. Stronger desktop references are the next approved experiment; compression follows an actual quality result.
