# TinySR technical context

## Current state — v1.0.0

Read this file before changing the implementation, then inspect the relevant source, tests, and linked technical evidence. Use the README for operational commands. Reconcile documentation with source rather than assuming either is universally authoritative. Update this active section when validation or the continuation point changes. Preserve important conditions and failure explanations; use versions and experiment identifiers rather than personal chronology.

TinySR is one project with desktop development tools and an OpenMV RT1062 deployment runtime. There are no permanent workflow tracks. Desktop work establishes model quality and performs training/conversion. The deployed camera must perform capture, face selection, ROI construction, inference, blending, and final-image creation. Host-generated review sheets are inspection artifacts, not deployed processing.

The source migration is complete for the capabilities listed in [migration.md](docs/migration.md). This is a refactor of working baselines, not a claim that the quality objective is solved. No new substantive training, powerful pretrained reference evaluation, physical camera experiment, or GPU benchmark was performed during migration.

### Validation boundaries

| Capability | Evidence status |
|---|---|
| Three recovered CNNs and six TFLite exports | Hash-verified; cleaned PyTorch artifacts preserve exact learned tensors |
| Same-scene native-scale evaluation | Executed with real INT8 artifacts and generated software witnesses |
| PyTorch/float-TFLite parity | Executed on deterministic non-face inputs; separate from real-scene quality |
| Tiling, residual scaling, review summaries, privacy allowlists, training resume | Behavioral regression coverage |
| Manual full-frame camera-local C16 output | Physically demonstrated by predecessor v6.3; new refactored revision awaits physical revalidation |
| Same-frame FOMO + C16 | Passed predecessor and migrated offline simulation; physical combined run remains pending |
| Camera visual quality | Development assessment found marginal improvement; aligned camera HR truth absent |
| Training | New trainer tested on tiny generated CPU data; substantive CUDA execution remains untested in this release |
| TensorFlow conversion | Channel-order regression and historical conversion evidence retained; new conversion execution requires the optional runtime and remains pending |
| Stronger reference model | Approved experiment plan, not implemented in v1.0.0 |

### Immediate continuation

1. Confirm installation and baseline audit on the execution machine using README steps; keep capability/environment output local.
2. Physically revalidate manual C16, automatic FOMO, no-face, and bicubic-only modes. Check that all outputs are camera-created and model hashes agree. Report unsupported APIs and allocation failures distinctly from detector negatives.
3. Implement the approved full-capacity desktop reference adapter and verified checkpoint acquisition described in [model_plan.md](docs/model_plan.md). Establish useful real-image quality before deciding whether downsizing is necessary.
4. Reuse development camera images for screening, then freeze a separate held-out collection before selecting deployment models.
5. Optimize the measured camera composition and transfer costs as separate changes with numerical controls. Do not infer speedups from file size or MAC counts alone.

### Important inherited findings

- The shallow CNNs use three 3×3 convolutions, one 1×1 head, and PixelShuffle ×4. Hidden width is 16, 24, or 32; the learned branch has a 7×7 LR receptive field. Its output is a normalized RGB residual, not final RGB.
- Add residual ×255 to the chosen byte-valued bicubic baseline. An earlier missing multiplier produced an effectively unchanged output. Changed pixels establish numerical effect, not visual improvement.
- Native-scale 64×64 inputs use three-pixel context and a 58-pixel valid step. Never downscale an arbitrary face into one tile merely to avoid tiled processing. Deeper networks need their own context analysis.
- A full 1280×960 RGB888 allocation failed in an earlier implementation. RGB565 uses fewer bytes and later succeeded, including coexistence with C16 and complete camera-local output. The RGB888 failure was not proof of total SDRAM exhaustion.
- FOMO supplies localization centers. The fixed head rectangle is a provisional heuristic, not measured head dimensions. No-face execution must remain explicitly bicubic-only.
- Native and converted Haar results are not interchangeable with desktop OpenCV results. Unsupported trees, missing files, invalid rectangles, and transport failures each require distinct interpretation.
- Host/device image paths differ in interpolation, quantized kernels, RGB565 storage and JPEG coding. Finished JPEG differences can extend beyond an edited ROI; raw-buffer invariants and compressed-file differences answer different questions.
- Checkpoint configurations can be top-level or nested in `frozen_identity`. Reject conflicts and load `model_state` strictly. Sanitized distribution checkpoints deliberately omit old optimizer/private run records.
- Original 100-image validation and 100-image test membership is preserved. Do not use the reserved neural test set for iterative model selection. The two training pools remain nested.
- Input photographs, their hashes, raw environment inventories, local paths, and original conversation context must remain outside tracked source and public exports.

## Project objective and decision rules

The deployment objective is a natural-looking complete 1280×960 image from a 320×240 camera frame, with face-aware neural processing and the rest of the field of view preserved. Post-capture 4× is the current task. Live output, optional fast/quality modes, and 8× are later decisions based on quality and measured resources.

Desktop model selection begins without an embedded storage ceiling. A useful model may already be feasible; if it is not, quantization, structured pruning, efficient redesign, or distillation can be investigated. Historical model storage and latency budgets were planning hypotheses, not design limits. Storage, activation/scratch memory, operator support, and measured end-to-end time are separate quantities.

Matched blind preference, artifact assessment, and synthetic reconstruction scores are complementary. The development-scene INT8 review found no model preference wins over bicubic; the original exporter merged ties and losses. The FP32 export contained pending ratings, while qualitative assessment still described marginal benefit. No claim of a completed blinded FP32 preference result follows. The unified summarizer now reports each outcome and direct model comparisons explicitly.

The provisional larger camera screen uses at least 20 held-out scenes and a 70% preference criterion with artifact review. It is an engineering decision rule, not a validated universal quality threshold. Repeated ratings of one scene do not create more independent scenes. Camera image differences cannot supply PSNR/SSIM without genuine aligned HR references.

## Maintaining this record

Describe evidence as hardware-measured, desktop-measured, simulated, derived, hypothetical, or planned. Keep measurements bound to model identity, version, firmware compatibility, input geometry, buffer lifetimes, and measured stage boundaries. Preserve significant rejected hypotheses and the observations that rejected them. Keep source, README, tests, and this active layer consistent.

This is a public technical record. Do not paste personal conversation history or environment dumps into it. Exclude personal names, dates, activity timestamps, host identities, specific individual machine inventories and local paths. Technical versions, update indices, duration measurements, anonymous scene IDs, public dataset IDs, and model hashes are appropriate when they explain reproducibility. Check serialized artifacts as well as visible prose before distribution.

## Evidence map

- [Architecture](docs/architecture.md): execution boundaries and module relationships
- [Model interface](docs/model_interface.md): layouts, residual units, quantization, tiling and detector semantics
- [Technical findings](docs/history/findings.md): controlled studies, hardware observations and failure explanations
- [Desktop model plan](docs/model_plan.md): quality-first RRDBNet experiment and later efficiency choices
- [Migration](docs/migration.md): preserved, consolidated and retired capabilities
- [Validation](docs/validation.md): executed checks and unresolved runtime gates
