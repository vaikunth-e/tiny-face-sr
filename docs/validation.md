# Validation — v1.0.0

## Executed migration checks

- **32 behavioral tests passed** using generated, nonprivate image data
- All nine distributed artifact hashes and byte lengths verified against the registry
- Exact equality of every learned tensor in all three cleaned PyTorch checkpoints against the recovered originals
- Metadata allowlist verified for distribution checkpoints; no optimizer, run identity, environment or private chronology remains
- PyTorch versus FP32 TFLite parity passed for all widths on black, midgray and seeded random non-face witnesses
- Maximum absolute normalized-residual differences were below **0.0000006**; INT8 execution and tensor contracts also passed
- Real INT8 baseline comparison produced nine tiles per model on a generated 140×140 ROI and preserved the raw desktop exterior exactly
- Automatic face, no-face, manual bypass and bicubic-only camera flows executed under a simulated OpenMV runtime
- Transfer fragmentation, corruption, duplicate starts and premature completion checks exercised
- Review wins/ties/losses, direct model pairs, missing rows and aggregate privacy exports checked
- Dataset membership confirmed: 5,000/20,000 nested training pools, common 100 validation and 100 reserved test IDs
- Four-update synthetic CPU training matched resumed versus uninterrupted learned tensors exactly; changed resume identity was rejected and a missing best-checkpoint derivative was recoverable
- PixelShuffle/DepthToSpace head channel ordering and Haar stump/tree rejection tested independently of TensorFlow
- Desktop source parsed; generated automatic/manual/bicubic/resource programs compiled
- Editable installation, command help, model audit and dataset audit executed
- Fresh Git clone independently passed the same 32 tests
- Wheel build succeeded and included maintained device sources plus shared geometry

These are software migration checks. They are not new real-camera quality results or substantive model-training outcomes. Integer stress-witness deviations are diagnostic observations, not an INT8 perceptual-quality gate.

## Distribution checks

The release tree is reviewed for private prose, dates, local paths, individual hardware inventories, image-derived fingerprints and unwanted artifacts. Checkpoint metadata is structurally checked rather than relying on text searches alone. The release archive excludes caches, installed package metadata, work/results directories, private source archives and inherited context files. Public text has no original personal timeline.

The initial Git commit uses a neutral project identity and synthetic epoch metadata, with no imported personal history or remote configuration. Source and device assets are included in one normal working repository. The archive uses neutral fixed entry timestamps, which do not represent development activity.

## Pending execution gates

- Physical RT1062 revalidation of the refactored commands and APIs
- Full same-frame automatic/no-face integration on actual hardware
- Repeated-capture memory recovery and fragmentation measurements
- Actual CUDA training and CUDA resume behavior in this revision
- New TensorFlow export execution, representative calibration and subsequent camera loading
- Installation across all advertised dependency ranges and operating systems
- Stronger reference-model checkpoint acquisition/inference and held-out visual evaluation

Historical physical measurements remain attached to predecessor versions in `history/findings.md`. They are not automatically transferred to this refactored release. No firmware, camera filesystem or existing local image dataset was changed during migration.
