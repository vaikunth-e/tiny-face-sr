# Migration record — v1.0.0

## Scope

This release replaces independently packaged desktop/model and camera programs with one maintained package. No original source archive, private context, transcript, personal environment dump, or legacy application tree is embedded. There is no contributor workflow document. Setup and patch mechanics are in the README; technical continuity is in CONTEXT.

The reviewed inputs were the latest camera v6.4 package with inherited manual/bicubic/resource controls, desktop baseline/evaluation v0.1.0, earlier v5.5.3 detector/integration source, and the reconstructed original training/conversion studies. `migration_ledger.json` records per-file disposition of relevant source inputs using family-relative names and hashes, without local roots or copied personal content.

## Consolidation map

| Prior responsibility | Maintained implementation | Disposition |
|---|---|---|
| Successive camera host runners | `camera/session.py`, `camera/results.py`, unified CLI | Consolidated; automatic/manual/bicubic modes |
| Board-source strings and versioned generators | `device/openmv/` and `camera/program.py` | Split into editable device modules; generated source stays local |
| Versioned image/report collectors | `camera/transport.py` | One framed collector; premature DONE now rejected |
| Repeated USBDBG clients | `camera/usbdbg.py` | One client; explicit local port required |
| Geometry copies and embedded tile loops | `common/geometry.py` | Shared source included by device assembler |
| Allocation diagnostic | `camera/resources.py`, device resource source | Preserved as a command using the same collector |
| Photo capture | `camera/capture.py` | Preserved as one QVGA acquisition command |
| Checkpoint loading/runtime adapters | `models/runtime.py`, `models/registry.py` | Separated from composition; strict supported schemas |
| Historical CNN definition | `models/residual.py` | One direct/residual topology; no copied reference model |
| Scene freeze and comparison | `data/scenes.py`, `evaluation/compare.py` | Portable CLI; legacy scene schema accepted |
| Blind review and export | `evaluation/review.py`, `evaluation/export.py` | Separate outcomes/direct pairs; allowlisted export |
| Dataset/study scripts | `data/pairs.py`, split manifests | Frozen membership retained; configuration rather than per-study programs |
| Training variants and continuation | `training/engine.py` | One configurable direct/residual trainer with exact new-format resume |
| Conversion bridge/audit | `conversion/tensorflow.py`, `conversion/export.py`, `conversion/verify.py` | Exact head mapping; representative INT8 inputs required |
| Structured pruning | `conversion/pruning.py` and CLI | Dense channel deletion retained; no quality/speed claim |
| Haar converter | `conversion/haar.py` | Retained as a checked library utility; tree rejection preserved |
| Face-box metrics | `evaluation/detection.py` | Retained independently of detector runtime |
| Multiple technical contexts | `CONTEXT.md`, `docs/history/findings.md` | Rewritten technical continuity; private originals excluded |

## Deliberately retired source

Old standalone launchers, per-version README/methodology copies, copied original source trees, packaging/repack scripts and temporary diagnostics are not carried into the maintained tree. Redundant tests tied to obsolete filenames or literal source strings were replaced with behavioral checks. Useful numerical, privacy, geometry, protocol, model and resume regressions remain.

The old streamed host-image reassembly path is no longer the deployed implementation. Its important numerical/transport evidence is preserved in technical history. Obsolete threshold sweeps, storage-root investigation scripts, unsupported detector profiles, historical experiment-specific report renderers and dataset download wrappers are retired; their findings, split definitions, model tensors, metric definitions and reusable conversion/scoring logic remain. The new package does not automatically reproduce every historical report layout or command name.

No dataset, local camera output, working camera filesystem, firmware, or original archive is modified by the migration. Existing scene/pair manifests remain readable locally. Physical automatic integration is still pending. The cleaned inference checkpoints intentionally cannot resume historical optimizer histories; the unified trainer has its own versioned resume contract.

## Artifact sanitization

The six baseline TFLite exports are byte-identical to the recovered files. Each PyTorch checkpoint is rewritten to contain only schema, model configuration, learned tensors and optimizer step. Tensor equality against the original was checked for all parameters. Distribution hashes therefore differ from original checkpoint hashes; both identities are retained in the registry. This is metadata sanitization, not retraining or a weight change.

The public split manifests contain only image IDs and split labels. Original personal chronology, configuration seeds that encoded calendar dates, paths and environment histories are excluded. New default experiment seeds are ordinary noncalendar integers. Existing model weights retain their original learned values.

## Remaining implementation gates

- Physical revalidation after restructuring, including manual, automatic, no-face and repeated-capture cases
- Substantive CUDA training and optional TensorFlow conversion execution in the unified revision
- A verified powerful reference model adapter and acquisition manifest
- Adaptive face-box/head sizing and broader detector evaluation
- Faster camera composition/transfer with correctness and buffer-lifetime controls

These are explicitly pending; offline migration checks are not substitutes for them.
