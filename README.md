# TinySR

TinySR develops face-aware 4× super-resolution from QVGA images and an OpenMV RT1062 camera implementation. Desktop tools cover data preparation, training, inference, comparison, and conversion. The camera implementation creates the complete output image on the device.

Version: **1.0.0**. Read [CONTEXT.md](CONTEXT.md) for current evidence and pending work. The recovered small models are functional baselines; meaningful visual improvement on real camera images remains an open objective. The stronger desktop model direction is recorded in [the model plan](docs/model_plan.md).

## 1. Install

Use Python 3.10 or newer. Python 3.12 was used for the migration checks. Run commands from the repository root. Windows PowerShell examples follow; on Linux or macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

```powershell
# create one desktop environment and install the editable package
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[camera,metrics,test]"
```

Install PyTorch separately using the CPU or CUDA wheel command appropriate for the machine from [PyTorch installation](https://pytorch.org/get-started/locally/). CUDA is preferred for substantive training. CPU inference and explicitly selected CPU training are supported. The camera and basic data commands do not need PyTorch.

For INT8/float-TFLite inference, install one supported interpreter. The tested Linux route is:

```powershell
# install the Linux LiteRT profile when compatible wheels are available
.venv/Scripts/python.exe -m pip install -r requirements/litert-linux.txt
```

On Windows use TensorFlow as the interpreter fallback:

```powershell
# install the optional TensorFlow interpreter and conversion dependencies
.venv/Scripts/python.exe -m pip install -e ".[convert]"
```

If conversion dependencies conflict with the CUDA environment, use a separate conversion environment with the same editable checkout. No second source tree is needed. Dependency ranges in `pyproject.toml` describe supported installation choices, not a claim that every version combination has been tested. [Validation](docs/validation.md) identifies executed checks.

```powershell
# inspect capabilities and verify all bundled artifact identities
.venv/Scripts/python.exe -m tinysr capabilities
.venv/Scripts/python.exe -m tinysr models audit

# compare PyTorch with float TFLite and exercise INT8 synthetic witnesses
.venv/Scripts/python.exe -m tinysr models verify --output work/runtime_verification.json
```

`models verify` needs PyTorch, NumPy, and a TFLite interpreter. It does not measure camera performance or perceptual image quality. Baseline assets are resolved relative to the checkout; set `TINYSR_ROOT` to that checkout if installing the package elsewhere. Keep private data and output under ignored `work/` or an external directory.

## 2. Run the camera pipeline

Use the OpenMV 4.5.9 compatibility baseline. Close other software holding the serial port. Obtain the actual camera port locally; replace `CAMERA_PORT` below. No port is hardcoded into host source.

The camera must already contain the exact C16 INT8 model. Copy `models/baselines/c16/residual_int8.tflite` to `/tinysr_rt1062/c16/residual_int8.tflite` on the camera filesystem, or supply its current on-device filename with `--model-path`. The program verifies its hash. It does not update firmware or overwrite `main.py`/`boot.py`.

```powershell
# use a manual source-coordinate ROI as the integration control
.venv/Scripts/python.exe -m tinysr camera run --port CAMERA_PORT --mode manual --manual 96 56 128 128 --output work/manual_v1

# run same-frame FOMO selection and C16 composition
.venv/Scripts/python.exe -m tinysr camera run --port CAMERA_PORT --mode auto --output work/automatic_v1

# execute the bicubic-only camera control
.venv/Scripts/python.exe -m tinysr camera run --port CAMERA_PORT --mode bicubic --output work/bicubic_v1
```

Inspect `work/<run>/local_only/camera_bicubic_4x.jpg`, `camera_final_4x.jpg`, and the comparison sheet. Both finished JPEGs come from the camera. The host verifies and saves them; the comparison sheet is a local review aid. Automatic mode uses a provisional 140-pixel head-size heuristic, configurable with `--head-size`. No face means explicit bicubic-only output and no C16 execution.

Camera output is 1280×960. The reference Python composition loop and paced USB transfer are slow; this is a post-capture experiment. The refactored camera code has passed offline simulation, but still needs physical revalidation. Earlier manual-mode hardware success does not establish physical success of this new revision or automatic mode.

```powershell
# acquire a saved QVGA photograph without neural processing
.venv/Scripts/python.exe -m tinysr camera capture --port CAMERA_PORT --output work/source_qvga.jpg

# inspect framebuffer allocation and optional C16 overlap
.venv/Scripts/python.exe -m tinysr camera resources --port CAMERA_PORT --with-c16 --output work/resources_v1.json
```

Resource reports and failure receipts stay local. `gc` counters are not a measurement of total physical peak RAM. `camera run` also creates an allowlisted `metrics_only.zip`; raw console, private paths, image data, and image fingerprints stay in `local_only/`.

## 3. Compare saved camera scenes

Create a CSV based on `configs/scenes_example.csv`. Use a real 320×240 source image, an anonymous scene ID, and a visually verified `(x,y,width,height)` ROI in source coordinates. Relative source paths resolve from the CSV directory. Mark `face_confirmed=yes` only after inspection. Optional camera bicubic/C16 controls need confirmation that they belong to the same source and ROI.

```powershell
# freeze source identity and source-coordinate ROI
.venv/Scripts/python.exe -m tinysr scenes --table work/scenes.csv --output work/scenes_v1.json

# evaluate the three baseline models on exactly the same inputs
.venv/Scripts/python.exe -m tinysr compare --scenes work/scenes_v1.json --backend int8 --output work/comparison_int8_v1

# run PyTorch FP32 with automatic CUDA or CPU selection
.venv/Scripts/python.exe -m tinysr compare --scenes work/scenes_v1.json --backend fp32 --device auto --output work/comparison_fp32_v1
```

Review the anonymous `pair_*.png` files under `local_only/<scene>/` before opening labeled outputs or the blind key. Fill `local_only/review.csv`: keep `pair_id`, supply an anonymous `rater_id`, set preference to `A`, `B`, or `tie`, and mark artifacts as `yes` or `no`. Leave incomplete reviews pending rather than fabricating ratings.

```powershell
# summarize every pair, including direct model comparisons
.venv/Scripts/python.exe -m tinysr review --results work/comparison_int8_v1

# export aggregates without local images, keys, paths, or machine inventories
.venv/Scripts/python.exe -m tinysr export --results work/comparison_int8_v1 --output work/comparison_metrics_v1.zip
```

Wins, ties, losses, pending rows, and artifact reports are separate. Scene-weighted rates prevent extra reviewers from making one scene count as more independent scenes. A one-scene development check is not a held-out quality evaluation. Without aligned HR camera truth, camera PSNR/SSIM is not established.

## 4. Prepare data and train

The two supplied FFHQ membership manifests preserve the original 5,000/20,000 training IDs and 100 validation/100 test IDs. No image dataset is bundled. Use an existing authorized local FFHQ collection; do not redistribute images through the repository.

```powershell
# check split separation before preparing or training
.venv/Scripts/python.exe -m tinysr data audit --manifest data/manifests/ffhq_20000.json

# prepare portable pairs from a source tree of ID-named PNGs
.venv/Scripts/python.exe -m tinysr data prepare --manifest data/manifests/ffhq_20000.json --source FFHQ_ROOT --output work/pairs_v1

# train a residual baseline with explicit CUDA execution
.venv/Scripts/python.exe -m tinysr train --config configs/residual_training.json --manifest work/pairs_v1/pairs.json --data-root work/pairs_v1 --device cuda --output work/training_v1
```

`FFHQ_ROOT` is a placeholder for a local path. Preparation uses Lanczos to 256×256 HR, then bicubic to 64×64 LR. Existing legacy pair manifests can instead be supplied directly with their original `--data-root`; source image paths do not need to be moved.

Copy and version the training configuration to change architecture, width, batch size, update budget, seed, or schedule. Explicit `--device cpu` permits slower CPU runs. A requested unavailable CUDA device fails without silent fallback. Run the same command to resume a new-format checkpoint; changed data, code, settings, or execution identity is rejected. `--stop-at` saves an intermediate update without changing the configured total budget. Historical optimizer checkpoints are not imported into the new trainer; inference weights are preserved.

The trainer currently supports the historical direct/residual topology and L1 objective. RRDBNet adaptation and perceptual training are planned in `docs/model_plan.md`, not implemented here. Validation uses the validation split; the test split remains reserved for final evaluation.

## 5. Prune or convert a baseline

```powershell
# create a dense channel-pruned candidate without claiming quality validation
.venv/Scripts/python.exe -m tinysr models prune --source C32 --keep 24 --output work/pruned_v1.pt

# export a supported residual checkpoint to float TFLite
.venv/Scripts/python.exe -m tinysr models convert --checkpoint models/baselines/c16/best.pt --output work/conversion_v1
```

For INT8 conversion add `--calibration work/calibration.npy`: a local normalized float RGB array `[N,64,64,3]` drawn from appropriate training/calibration inputs, not the held-out test set. Do not commit this image-derived array. PixelShuffle channel order is explicitly mapped to TensorFlow DepthToSpace. An export is not proof of OpenMV operator support, perceptual quality, or speed. Pruning requires subsequent fine-tuning and evaluation; sparse zeros alone do not imply acceleration.

Reusable face-box scoring and the checked Haar binary converter remain available in the Python package. `python -m tinysr score-detections --help` describes JSON scoring inputs. FOMO center heuristics cannot be scored as learned face bounds. See `docs/model_interface.md` for contracts.

## 6. Check and update the repository

```powershell
# execute synthetic, protocol, scientific and resume regressions
.venv/Scripts/python.exe -m pytest -q
```

The full suite needs PyTorch, scikit-image, and a supported TFLite runtime. Physical camera and CUDA validation are separate. Tests use generated images and never need private camera photographs.

If receiving a unified patch, first check its stated base revision and local changes. From a clean checkout:

```powershell
# validate then stage additions, modifications and deletions
git apply --check --index ../tinysr-update.patch
git apply --index ../tinysr-update.patch
git diff --cached --stat
```

Run the checks supplied with the patch before committing. If the base differs, resolve the change deliberately rather than forcing it. A normal revert can undo a committed update without deleting local datasets or results. This describes patch mechanics; branch/review conventions belong to the collaborating group.

The repository contains no contributor workflow document. Technical state lives in `CONTEXT.md`; detailed evidence lives in `docs/history/`. Remove obsolete source and temporary investigations after preserving their technical findings. Retain useful regression coverage. Keep personal records, dates, local settings and environment inventories out of tracked files and release contents.

## Source map

| Directory | Responsibility |
|---|---|
| `src/tinysr/camera/` | Host control, transport, capture and temporary-program assembly |
| `src/tinysr/device/openmv/` | Maintained camera-runtime source modules |
| `src/tinysr/common/` | Shared geometry, atomic records and capability selection |
| `src/tinysr/models/` | Baseline architecture, adapters and immutable registry access |
| `src/tinysr/data/`, `training/` | Frozen datasets, preparation and resumable training |
| `src/tinysr/evaluation/`, `conversion/` | Composition, metrics, review, export and compression utilities |
| `models/`, `data/manifests/` | Cleaned baseline artifacts and split membership |
| `tests/`, `docs/` | Behavioral regressions and technical documentation |

The camera sources live inside the package so editable installs and built distributions assemble the same maintained files. Generated launch programs are local output, not a second implementation to edit.
