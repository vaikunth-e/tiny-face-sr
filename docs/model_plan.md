# Desktop quality-model plan — v1

**Status:** Approved research direction; reference inference and fine-tuning are not implemented by the repository refactor.

Establish a model that visibly improves real camera images on a desktop without an initial embedded model-storage ceiling. Measure resource use, but defer camera feasibility and compression decisions until a useful quality result exists. CUDA is preferred where available; portable inference should remain possible on CPU.

Start from RRDBNet with Real-ESRGAN x4plus perceptual weights and RealESRNet x4plus reconstruction-oriented weights. The planned configuration has 64 feature channels, 23 residual-in-residual dense blocks and growth width 32. It extracts features, processes a deep dense/residual trunk, upsamples twice by 2×, and predicts RGB directly. It must have its own adapter, not bicubic-residual postprocessing.

Acquire official checkpoints, retain upstream licenses and record hashes before treating results as reproducible. The first comparison uses bicubic, recovered C16 FP32, RealESRNet and Real-ESRGAN on exactly the same frozen development image. Prefer whole-frame 320×240 inference to 1280×960 and inspect matching face regions at equal display sizes. Keep optional face restoration disabled initially. If required, validate tiled inference and its context against whole-image inference before comparing quality.

| Stage | Intervention | Decision |
|---|---|---|
| Reference | Official pretrained inference | Is useful improvement already available? |
| Domain adaptation | Camera-like/native-scale data with L1 | Does application matching help fidelity? |
| Perceptual branch | Same parent/data/update budget plus perceptual loss | Does detail improve without objectionable artifacts? |
| Optional adversarial branch | Separately justified fine-tuning | Is texture worth the artifact risk? |

Preserve frozen FFHQ partitions and keep the test split out of model selection. Development scenes used to choose architecture/settings cannot become unseen confirmation. Evaluate selected settings on a separately frozen camera set. Report wins/ties/losses, artifacts, facial geometry, hair/skin/eye detail and failure examples. Synthetic PSNR/SSIM is complementary; absent aligned camera HR, generated texture is not measured recovery.

A comparison of pretrained systems does not isolate loss-function effects because training histories differ. Domain degradation recipes are hypotheses. Change one factor at a time where causal interpretation matters. Profile data loading, transfer, training and validation separately; using CUDA alone does not guarantee wall-clock improvement.

Once quality is useful, measure file storage, live tensor/buffer memory, operators and actual camera latency. A practical model can be deployed directly. Otherwise evaluate INT8/PTQ/QAT, dense channel pruning plus fine-tuning, efficient architecture changes, or distillation. Pruning is one tool and does not guarantee an arbitrary model can fit. Unstructured zeros do not imply acceleration with dense kernels.
