# Unsupervised Industrial Bottle Defect Detection Using a Convolutional VAE

Research question (unchanged):

> Can a variational autoencoder trained only on defect-free bottle images detect
> previously unseen manufacturing defects using reconstruction error?

Phase 1 sets up the repository, Python environment and dataset checks.
Phase 2.1 inspects and prepares normal training images, starting with E1.
Phase 2.2 implements the E1 convolutional VAE and checks its forward pass on real
normal training images. Phase 2.3 implements the VAE loss, a normal-only
training/validation split and E1 training. The initial three-epoch run and the full
50-epoch E1 run have completed. Phase 2.4 inspects the recorded losses and
reconstructions of held-out normal images. Phase 3's reconstruction-error scoring,
normal-validation threshold calibration and three normal-image heatmaps are
complete for the E1 baseline. Phase 4's configuration runs and normal-image
outputs are complete. Phase 5 image-level evaluation is now complete on all 83
actual reserved test images, using frozen checkpoints and thresholds. E2 preprocessing,
model support and reuse of E1's split are verified. E2 completed 50 training
epochs; its selected checkpoint is epoch 45. E2 reconstruction inspection,
normal-validation calibration and heatmaps are now complete. E3 preparation and
implementation are verified. E3 training stopped early after 23 epochs; its
selected checkpoint is epoch 13. E3 normal-image inspection, calibration and
heatmaps are now complete.
Training uses only normal training images. Test results must not be used to
retune the models or thresholds. Ground-truth masks were not used in Phase 5.
Earlier sections describe progress at their recorded stages; the Phase 5 section
below contains the reserved test results.

## Original six-phase order and current progress

Phases are project stages; E1, E2 and E3 are experiment configurations.

| Phase | Scope | Current progress |
| --- | --- | --- |
| 1 | Setup and dataset checks | Complete locally; actual counts and mask matches documented |
| 2 | Build and train the baseline VAE: 2.1 preparation, 2.2 model, 2.3 loss/training, 2.4 inspection | Complete for the E1 baseline |
| 3 | Implement anomaly scores and heatmaps | E1 image scoring, normal-validation threshold calibration and requested heatmaps complete |
| 4 | Run E1, E2 and E3 consistently; record outputs and computational cost | Complete for the agreed E1/E2/E3 configurations: training, normal-image inspection/calibration/heatmaps, parameter counts and training runtimes recorded |
| 5 | Evaluate defect-detection performance | Complete for image-level evaluation: the same 83 test images scored by E1/E2/E3 with frozen thresholds; confusion matrices, category totals and ROC-AUC saved |
| 6 | Complete the research paper and demo | Outstanding; documentation and figures are available for later use |

Phases 4 and 5 are complete for the agreed configurations and image-level
evaluation. The next unfinished phase is Phase 6, the paper and demo, which has
not started. Do not choose or retune thresholds using the test results.

## Revised experiment plan

The original proposal specified comparing latent dimensions of 16, 32 and 64.
The experimental method has deliberately been revised to vary image resolution
and latent dimension together:

| Experiment | Input resolution | Latent dimension |
| --- | --- | ---: |
| E1 | 128 × 128 | 16 |
| E2 | 256 × 256 | 32 |
| E3 | 512 × 512 | 64 |

This supersedes the earlier suggestion of four resolutions with a fixed latent
dimension of 32. The research question remains unchanged, and the original
proposal files remain unchanged.

The Phase 5 results below compare detection performance and recorded training cost.
Because both settings change together, the comparison concerns whole
configurations: resolution, latent dimension and resulting model parameter count
all change. It cannot isolate any one of these effects. Any observed winner is
described as **best among the
configurations tested**, not universally optimal.

Image resolution is the number of pixel rows and columns supplied to the model;
an RGB image also has three colour channels. Latent dimension is the number of
values in the model's learned compressed representation. It is recorded here for
the model phase and does not affect preprocessing.

## Files

```text
bottle-vae-anomaly-detection/
├── README.md
├── AGENTS.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/                       # ignored
│   └── bottle/
│       ├── readme.txt
│       ├── license.txt
│       ├── train/good/
│       ├── test/
│       │   ├── good/
│       │   ├── broken_large/
│       │   ├── broken_small/
│       │   └── contamination/
│       └── ground_truth/
│           ├── broken_large/
│           ├── broken_small/
│           └── contamination/
├── src/bottle_vae/
│   ├── __init__.py
│   ├── data_check.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── model_check.py
│   ├── loss.py
│   ├── train.py
│   ├── inspect_training.py
│   ├── scoring.py
│   ├── heatmaps.py
│   └── evaluate.py
├── notebooks/
│   └── .gitkeep
├── tests/
│   └── .gitkeep
└── outputs/                    # ignored; generated reports, training results and previews
    ├── E1/
    │   ├── preprocessing_report.json
    │   ├── model_check_report.json
    │   ├── initial_training/
    │   │   ├── data_split.json
    │   │   ├── history.csv
    │   │   ├── best_model.pt
    │   │   └── training_summary.json
    │   └── full_training/
    │       ├── data_split.json
    │       ├── history.csv
    │       ├── best_model.pt
    │       ├── training_summary.json
    │       ├── inspection/
    │       │   ├── loss_curves.png
    │       │   ├── loss_curves.pdf
    │       │   ├── 001_reconstruction.png
    │       │   ├── 087_reconstruction.png
    │       │   ├── 206_reconstruction.png
    │       │   └── inspection_report.json
    │       ├── calibration/
    │       │   ├── validation_scores.csv
    │       │   └── threshold.json
    │       ├── heatmaps/
    │       │   ├── 001_heatmap.png
    │       │   ├── 087_heatmap.png
    │       │   ├── 206_heatmap.png
    │       │   └── heatmap_report.json
    │       └── evaluation/
    │           ├── dataset_report.json
    │           ├── image_scores.csv
    │           ├── metrics.json
    │           ├── provenance.json
    │           ├── roc_curve.csv
    │           └── roc_curve.png
    ├── E2/
    │   ├── preprocessing_report.json
    │   ├── model_check_report.json
    │   ├── training_configuration.json
    │   ├── data_split.json
    │   ├── split_verification.json
    │   └── full_training/
    │       ├── data_split.json
    │       ├── history.csv
    │       ├── best_model.pt
    │       ├── training_summary.json
    │       ├── inspection/
    │       │   ├── loss_curves.png
    │       │   ├── loss_curves.pdf
    │       │   ├── 001_reconstruction.png
    │       │   ├── 087_reconstruction.png
    │       │   ├── 206_reconstruction.png
    │       │   └── inspection_report.json
    │       ├── calibration/
    │       │   ├── validation_scores.csv
    │       │   └── threshold.json
    │       ├── heatmaps/
    │       │   ├── 001_heatmap.png
    │       │   ├── 087_heatmap.png
    │       │   ├── 206_heatmap.png
    │       │   └── heatmap_report.json
    │       └── evaluation/
    │           ├── dataset_report.json
    │           ├── image_scores.csv
    │           ├── metrics.json
    │           ├── provenance.json
    │           ├── roc_curve.csv
    │           └── roc_curve.png
    ├── E3/
    │   ├── preprocessing_report.json
    │   ├── model_check_report.json
    │   ├── training_configuration.json
    │   ├── data_split.json
    │   ├── split_verification.json
    │   └── full_training/
    │       ├── data_split.json
    │       ├── history.csv
    │       ├── best_model.pt
    │       ├── training_summary.json
    │       ├── inspection/
    │       │   ├── loss_curves.png
    │       │   ├── loss_curves.pdf
    │       │   ├── 001_reconstruction.png
    │       │   ├── 087_reconstruction.png
    │       │   ├── 206_reconstruction.png
    │       │   └── inspection_report.json
    │       ├── calibration/
    │       │   ├── validation_scores.csv
    │       │   └── threshold.json
    │       ├── heatmaps/
    │       │   ├── 001_heatmap.png
    │       │   ├── 087_heatmap.png
    │       │   ├── 206_heatmap.png
    │       │   └── heatmap_report.json
    │       └── evaluation/
    │           ├── dataset_report.json
    │           ├── image_scores.csv
    │           ├── metrics.json
    │           ├── provenance.json
    │           ├── roc_curve.csv
    │           └── roc_curve.png
    ├── phase5_comparison.csv
    └── preprocessing/
        └── <image stem>_comparison.png
```

- README.md explains scope, setup, data and commands.
- AGENTS.md records project working preferences, including disclosure and cleanup
  of any temporary code or data used for a requested check.
- requirements.txt lists Pillow for image reading/resizing and NumPy for pixel
  arrays and numerical checks, PyTorch for the VAE, and Matplotlib for training
  loss plots. Phase 1 itself uses only the standard library.
- pyproject.toml defines package metadata, the src layout, Python >=3.10,
  and an optional installed command. Setuptools is only a packaging dependency.
- .gitignore excludes the dataset, images, archives, virtual environments,
  caches, credentials and generated results.
- src/bottle_vae/__init__.py marks the Python package.
- src/bottle_vae/data_check.py checks required folders, PNG file counts,
  PNG signatures and one-to-one defect image/mask filename matches.
- src/bottle_vae/preprocessing.py inspects normal training images, prepares and
  checks one image at a time, and optionally saves labelled visual comparisons.
- src/bottle_vae/model.py defines the configurable four-block VAE: E1 defaults
  are 128 × 128 RGB / latent 16; E2 uses 256/32 and E3 uses 512/64.
- src/bottle_vae/model_check.py checks the untrained model on real normal training
  images and writes a report. For E2/E3 it also verifies split reuse and loss
  reporting. E3 additionally checks the full batch of eight and both saved
  E1/E2 checkpoints. Shared training settings are checked against saved runs.
  It remains a
  development helper scheduled for final cleanup.
- src/bottle_vae/loss.py computes total, reconstruction and KL losses for training.
- src/bottle_vae/train.py loads normal images, can fit E1, E2 or E3, measures normal
  validation loss, saves a best checkpoint and records settings, filenames,
  parameter count, loss history and runtime. E2/E3 load E1's saved filename split.
- src/bottle_vae/inspect_training.py plots a saved run's losses and compares three
  held-out normal images with reconstructions from the selected checkpoint.
  E2/E3 reuse the source paths recorded by E1's inspection when a trained checkpoint exists.
- src/bottle_vae/scoring.py computes one reconstruction-error score per image
  and records each configuration's own threshold from its normal validation split.
  Its shared `load_model` function strictly loads saved weights and checks
  dimensions/selected epoch for calibration and test evaluation.
- src/bottle_vae/heatmaps.py uses the same three inspected normal images and the
  recorded checkpoint to compute spatial errors, verify score equivalence and
  save labelled heatmaps with one shared numerical colour scale.
- src/bottle_vae/evaluate.py verifies the actual reserved test images, reuses
  `load_model` and `score_images`, applies frozen thresholds and saves image-level
  confusion matrices, category results, tied-score-aware ROC/AUC and comparison
  data. It never recalibrates thresholds or reads masks.
- tests/.gitkeep preserves the empty tests directory; no synthetic tests are included.
- notebooks/.gitkeep keeps an empty notebook directory in Git.
- data/ holds the original local data. outputs/ holds generated reports and previews.
  These directories are not distributed through Git and must be created after cloning.

## Windows PowerShell setup

Run from the repository root, using Python 3.10 or newer:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, outputs | Out-Null
```

Activation is optional: calling the environment's interpreter directly avoids
PowerShell execution-policy changes. To activate when your policy allows it:

```powershell
.\.venv\Scripts\Activate.ps1
# When finished:
deactivate
```

Obtain the bottle archive from the official MVTec page below and extract its
bottle directory into data/, preserving the original readme.txt and license.txt.
This checkout already has data/bottle; do not extract another nested copy.
For a fresh checkout with bottle.tar.xz in the parent directory:

```powershell
tar -xf ..\bottle.tar.xz -C data
```

## Run the Phase 1 dataset check

This standard-library checker needs no package installation or network access:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\data_check.py --data-root .\data\bottle
$LASTEXITCODE
```

The root argument must point to the directory containing train, test and
ground_truth. Relative paths are resolved from the current working directory.
Exit 0 means every Phase 1 check passed; exit 1 reports missing/empty folders,
unexpected entries, unreadable files, bad PNG signatures, mismatched masks or
proposal count differences. Missing folders are printed as MISSING, not zero.
The checker reads only PNG signatures, not decoded pixels, dimensions or image
content; success does not establish that every image fully decodes.

Optional editable installation exposes a command and module entry point
(this may download the setuptools build dependency):

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\bottle-data-check.exe --data-root .\data\bottle
.\.venv\Scripts\python.exe -m bottle_vae.data_check --data-root .\data\bottle
```

Dataset validation uses the actual project images in data/bottle. The checker
only reads folders and files. If temporary code or data is needed for a requested
check, its use must be explained beforehand and its deletion confirmed afterward.

## Dataset and proposal counts

Normal training: proposal 209. Normal test: proposal 20.
Defective test total: proposal 63. These are comparison targets, not assumed facts.
The executable checker reports observed counts on every run.
Defect masks are counted separately from test images.
A defect image such as test/broken_large/000.png must have
ground_truth/broken_large/000_mask.png. Normal images do not need masks.

Checked on 2026-09-26 with Python 3.13.2: the real dataset passed all Phase 1
checks.

| Folder | Actual PNG count |
| --- | ---: |
| train/good | 209 |
| test/good | 20 |
| test/broken_large | 20 |
| test/broken_small | 22 |
| test/contamination | 21 |
| ground_truth/broken_large | 20 |
| ground_truth/broken_small | 22 |
| ground_truth/contamination | 21 |

Defective test total: 63; total input images: 292; masks: 63.
All three proposal totals match. Missing required folders: none.
All 63 defect images have matching masks, with no orphan masks.
This is structural validation, not full image decoding or model evaluation.

## Phase 2.1: image inspection and preparation

Only `data/bottle/train/good/` is read during this phase. Test images and masks are
not used. Originals are never modified, and no duplicate prepared dataset is
saved. Each image is processed when loaded, one image at a time.

The script checks every file in the training folder, reporting actual image
count, original dimensions, colour modes, NumPy pixel data types and observed
pixel range. The expected count of 209 and dimensions of 900 × 900 are comparison
targets, not values substituted for the observations. Unreadable or corrupted
files and other discrepancies are reported explicitly.

`preprocess_image(path, resolution=128)` performs these steps:

1. Open the original image and convert it to RGB.
2. Resize directly from the original to 128, 256 or 512 pixels square using
   Pillow's bilinear resampling for every configuration.
3. Convert the pixels to a NumPy `float32` array and divide by 255.
4. Return the array in height × width × channels order.

Resizing changes spatial detail. Dividing by 255 changes the numerical scale
from `[0, 255]` to `[0, 1]`; it does not restore detail lost through resizing.
For example, RGB `(255, 128, 0)` becomes approximately
`(1.0, 0.502, 0.0)`. These are colour intensities, not defect probabilities.

### PowerShell commands

Run from the repository root. Phase 2.1 uses Pillow and NumPy; the current
requirements also include PyTorch for Phase 2.2. With the environment set up as
above, install any missing project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

E1 preparation checks all normal training images at 128 × 128 and creates the
three visual comparisons:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\preprocessing.py --experiment E1 --previews
$LASTEXITCODE
```

The same preparation can be run for E2 and E3:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\preprocessing.py --experiment E2
.\.venv\Scripts\python.exe .\src\bottle_vae\preprocessing.py --experiment E3
```

The default dataset root is `data/bottle`; `--data-root .\data\bottle` can also be
supplied explicitly. The script always uses `train/good` beneath that root.
Reports go to `outputs/E1/preprocessing_report.json`,
`outputs/E2/preprocessing_report.json` or `outputs/E3/preprocessing_report.json`
according to the selected configuration. These commands perform **preprocessing
checks**, not completed model experiments.

### Inspection results and previews

Checked on 2026-09-26 using the actual `data/bottle/train/good/` images with
Python 3.13.2, Pillow 12.3.0 and NumPy 2.5.3:

| Check | Observed result |
| --- | --- |
| Original images read successfully | 209 of 209 |
| Original dimensions | 900 × 900 for all 209 |
| Original colour mode | RGB for all 209 |
| Original pixel data type | `uint8` for all 209 |
| Original pixel range across all images | 23 to 255 |
| Unreadable/corrupted files or inspection discrepancies | None |
| E1 images processed and verified | 209 of 209 |
| E1 output shape and data type | `(128, 128, 3)`, `float32` for all 209 |
| E1 finite values within `[0, 1]` | Passed for all 209 |
| E1 observed range across all prepared images | Approximately 0.117647 to 1.0 |
| E1 preparation failures | None |

The E1 preprocessing check exited successfully. Its observed minimum reflects
bilinear resizing followed by division by 255; bilinear interpolation can change
the extrema by blending neighbouring pixel values. Scaling bounds the values to
`[0, 1]` but does not require any image to contain both 0 and 1.

The saved report is `outputs/E1/preprocessing_report.json`. No temporary scripts
or synthetic data were used. E2's full 209-image preparation check was completed
on 2026-10-06. E3's full 209-image preparation check was completed on 2026-10-07;
see the Phase 4 sections below. The earlier three comparison previews remain unchanged.

With `--previews`, the script selects three evenly spaced images from the sorted
filenames. This run selected `000.png`, `104.png` and `208.png`. Each comparison
shows the original alongside versions resized directly from it to 128, 256 and
512. The overview panels are displayed at 320 × 320 using bilinear resampling;
labels distinguish each image's actual resolution from its displayed size.
Each panel also shows a crop around the upper bottle rim: x from 3/8 to 5/8 and
y from 1/8 to 3/8 of the image, rounded down to pixel boundaries. These crops are
displayed at 320 × 320 using nearest-neighbour enlargement, which reveals the
pixel detail without adding new detail.

The saved previews are `outputs/preprocessing/000_comparison.png`,
`outputs/preprocessing/104_comparison.png` and
`outputs/preprocessing/208_comparison.png`.
Open the output folder in Windows File Explorer to inspect them:

```powershell
Invoke-Item .\outputs\preprocessing
Get-Content .\outputs\E1\preprocessing_report.json
```

Reports and previews remain excluded from Git by the existing `outputs/` rule.
Visual comparisons show changes in image detail; they cannot establish
defect-detection performance. Model evaluation comes later.

For the model phase: sigmoid can constrain each reconstructed RGB channel
independently to `[0, 1]`. Softmax imposes a sum-to-one constraint that RGB
intensities do not require. Reconstruction error is an anomaly score, not
automatically a confidence probability.

## Phase 2.2: E1 convolutional VAE

The baseline described here uses 128 × 128 RGB images and latent dimension 16.
`BottleVAE(image_size=128, latent_dim=16)` retains these defaults; `E1VAE` remains
a compatibility name for the same class, with unchanged checkpoint parameter
keys and shapes at the defaults. Phase 4 adds E2 through the same class using
`image_size=256, latent_dim=32`, and E3 with `image_size=512, latent_dim=64`.
E3's implementation is verified; its actual training results are recorded below.

PyTorch expects image tensors in batch × channels × height × width order.
The existing preprocessing function returns height × width × channels, so
`torch.from_numpy(image).permute(2, 0, 1)` changes `(128, 128, 3)` to
`(3, 128, 128)`. Stacking images adds the batch dimension. Inputs are `float32`
with values in `[0, 1]`; the channel order is RGB.

### Architecture and shape changes

Here, `B` is the number of images in a batch. All convolution and transposed
convolution layers use kernel size 4, stride 2 and padding 1. These settings halve
the height and width at each encoder stage and double them at each decoder stage.
See the PyTorch [Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
and [ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html)
API documentation for the shape rules.

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | Prepared RGB images | `(B, 3, 128, 128)` |
| Encoder 1 | Conv2d, 3 → 16 channels, ReLU | `(B, 16, 64, 64)` |
| Encoder 2 | Conv2d, 16 → 32 channels, ReLU | `(B, 32, 32, 32)` |
| Encoder 3 | Conv2d, 32 → 64 channels, ReLU | `(B, 64, 16, 16)` |
| Encoder 4 | Conv2d, 64 → 128 channels, ReLU | `(B, 128, 8, 8)` |
| Flatten | 128 × 8 × 8 feature values | `(B, 8192)` |
| Latent parameters | Separate linear layers for mean and log variance | Two `(B, 16)` tensors |
| Latent vector | Sample during training mode; use mean by default in evaluation mode | `(B, 16)` |
| Decoder input | Linear 16 → 8192, ReLU, reshape | `(B, 128, 8, 8)` |
| Decoder 1 | ConvTranspose2d, 128 → 64 channels, ReLU | `(B, 64, 16, 16)` |
| Decoder 2 | ConvTranspose2d, 64 → 32 channels, ReLU | `(B, 32, 32, 32)` |
| Decoder 3 | ConvTranspose2d, 32 → 16 channels, ReLU | `(B, 16, 64, 64)` |
| Reconstruction | ConvTranspose2d, 16 → 3 channels, sigmoid | `(B, 3, 128, 128)` |

Four encoder stages give a straightforward progression from pixels to smaller
feature maps; increasing the channel count allows more feature types at each
stage. The decoder mirrors those stages to restore the required image size.
ReLU provides a simple nonlinearity between layers. This is a small baseline
chosen for clarity and manageable computation, with no claim that its depth or
channel counts are optimal. The final sigmoid scales reconstructed channels
independently into `[0, 1]`, matching preprocessing.
The E1 model has 747,395 trainable parameters. Its original architecture check
used untrained weights; the first training run is recorded below.

### What the model methods do

- `encode` converts each input image into two 16-value vectors: the mean and log
  variance of its latent distribution. These are 16 latent dimensions, not 32.
- `reparameterize` computes `z = mean + exp(0.5 * log_variance) * noise`, where
  `noise` follows a standard normal distribution. This standard VAE sampling
  step lets gradients later flow through the mean and standard deviation; see
  the [official PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py).
- `decode` converts the 16-value latent vector into an RGB reconstruction.
- `forward` connects these steps and returns `(reconstruction, mean, log_variance)`.

In training mode, `forward` samples the latent vector. In evaluation mode
(`model.eval()`), it uses the mean by default, giving repeatable output for a fixed
input and fixed parameters on this CPU check. `sample=True` or `sample=False`
overrides this choice. Calling training mode does not train the model by itself;
learning needs a loss, gradients and parameter updates, which are not run here.

The random latent noise is part of VAE mathematics. It does not create synthetic
bottle images or replace the real dataset. The architecture is our E1 baseline;
the linked PyTorch example is a reference for sampling, not the source of this
entire architecture.

### Run the real-image model check

The current Windows environment uses Python 3.13.2 and this check runs on the CPU.
Install the CPU PyTorch package and the remaining requirements in the existing
environment, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\src\bottle_vae\model_check.py
$LASTEXITCODE
Get-Content .\outputs\E1\model_check_report.json
```

The development check reads the first, middle and last sorted image filenames from
`data/bottle/train/good/`, using the existing 128 × 128 preprocessing. It checks
a three-image batch and a single-image batch, including input/output shapes,
data types, finite values, reconstruction range, deterministic evaluation and
latent sampling. It also checks that model parameters remain unchanged.

The report is saved to `outputs/E1/model_check_report.json`, which is excluded
from Git. No temporary scripts, synthetic datasets, trained checkpoints or
reconstruction pictures are created by this check. The original images remain
unchanged, and no test images or masks are read.

Checked on 2026-10-06 with Python 3.13.2, PyTorch `2.14.1+cpu` and random seed 42.
The real source images were `000.png`, `104.png` and `208.png` in
`data/bottle/train/good/`. All 13 checks passed with no failures:

| Check | Verified result |
| --- | --- |
| Three-image input and reconstruction | `(3, 3, 128, 128)` |
| Encoder feature maps | `(3, 128, 8, 8)` |
| Mean, log variance and sampled latent vectors | `(3, 16)` each |
| Single-image reconstruction | `(1, 3, 128, 128)` |
| Tensor data types and finite values | All checked tensors `float32` and finite |
| Input and reconstruction intensities | Within `[0, 1]` |
| Evaluation using the mean | Repeatable; explicit `sample=False` also passed |
| Latent sampling | Two samples differed |
| Model parameters | Unchanged; zero optimizer steps |

This forward-pass check establishes that the untrained E1 model runs and produces
the expected tensor structures. It does not establish reconstruction quality or
defect-detection performance. No loss, backward pass, training, anomaly scores or
checkpoints were produced. The report is the record of these architecture checks.

The loss and first training run are documented below. Defect-detection evaluation
remains a separate step.

At final project cleanup, standalone check scripts and verification-only code
will be removed, with the working preprocessing, model, training and evaluation
pipeline kept intact. Temporary verification scripts used during development
must be disclosed before use, deleted afterward, and their deletion confirmed.

## Phase 2.3: VAE loss and E1 training

`vae_loss(reconstruction, images, mean, log_variance)` in `loss.py` returns three
differentiable scalar tensors: total loss, reconstruction loss and KL loss.

- Reconstruction loss encourages the decoder's output to match the input image.
  For each image, compute half the sum of squared errors over all pixels and RGB
  channels, then average those sums across the batch.
- KL loss encourages each encoded diagonal Gaussian distribution to stay close
  to a standard normal prior. Sum the contributions over latent dimensions for
  each image, then average across the batch.
- Total loss is the sum of these two batch averages, with a KL weight of 1.

For an image `x`, reconstruction `r`, mean `mu` and log variance `logvar`:

```text
reconstruction per image = 0.5 * sum((r - x)^2)
KL per image             = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
total loss               = mean(reconstruction per image) + mean(KL per image)
```

The reconstruction term assumes independent Gaussian observation noise with
fixed variance 1; the constant term is omitted because it does not affect
gradients. This is a deliberate initial modelling choice. The variational
objective and Gaussian KL formulation are described in
[Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/html/1312.6114v11).

This reconstruction reduction is different from averaging squared error over
pixels. E1 has `128 * 128 * 3 = 49,152` colour values per image, so its reconstruction
loss equals `24,576 * pixelwise MSE`. Both terms are averaged across images;
changing the batch size alone does not multiply the objective. Raw total losses
at different resolutions should not be used to rank E1, E2 and E3 because the
pixel count changes. Detection performance will require separate evaluation.

For E2, `256 * 256 * 3 = 196,608`, so `R = 98,304 * pixel-averaged RGB MSE`.
At equal pixel-averaged MSE, E2's reconstruction term is four times E1's.
At equal KL per latent dimension, summing 32 rather than 16 dimensions doubles
the KL term. Under those conditions the relative reconstruction/KL weighting
doubles. These scaling statements do not predict the losses a trained model
will actually achieve. We retain the agreed objective `total = R + K`, without
rescaling either term, and compare complete configurations with this limitation.

New runs additionally record two quantities for both fitting and validation:

```text
reconstruction_mse = 2 * R / (3 * H * W)
kl_per_dimension  = K / latent_dimension
```

These are reporting quantities only; gradients, checkpoint selection and early
stopping still use `R + K`. Like the existing losses, they are averaged over
images, weighting each batch by its image count. E1's saved histories are kept
unchanged; its normalized quantities can be calculated from the recorded R and K.

### Actual loss verification

On 2026-10-06, a disclosed temporary script used actual normal images `000.png`,
`104.png` and `208.png` from `data/bottle/train/good/`, with random seed 42.
The loss matched independent PyTorch reference calculations. Duplicating the
same real-image batch preserved the averaged losses, and using an image as its
own reconstruction gave zero reconstruction loss.

The reconstruction gradient reached the encoder's log-variance head through
latent sampling. All 22 model parameter tensors had finite gradients, and their
values stayed unchanged. No optimizer steps or checkpoints were produced.
The untrained batch gave approximately 3205.9807 total loss, 3205.9792 reconstruction
loss and 0.001462 KL loss; these verify the calculation and do not measure trained
reconstruction quality or defect-detection performance.

The temporary script `outputs/_phase23_loss_check.py` was deleted after use.
No temporary datasets or synthetic input images were created.

### E1 training settings

`train.py` uses the following starting settings. The initial run requested only
three epochs; the default command allows up to 50 with early stopping.

| Setting | Starting value |
| --- | --- |
| Data used for fitting and validation | Only `data/bottle/train/good/` |
| Normal training/validation split | 167 fitting images and 42 validation images |
| Random seed | 42; save the selected filenames for reproducibility |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 8 |
| Maximum epochs | 50 |
| Early stopping | Stop after 10 epochs without validation improvement |

This 80/20 division is within the 209 normal training images. It does
not change MVTec's supplied training/test split. The 20 normal and 63 defective
test images remain reserved for final evaluation. Use the same normal split
across later configurations so the comparison is consistent.

During fitting, sample the latent vector and update weights using the total loss.
Validation uses no weight updates. For a validation objective consistent with
the sampled VAE loss, sampling remains explicit even in evaluation mode; mean-only
decoding is a separate repeatable reconstruction convention, not an estimate of
the full variational objective. Record reconstruction and KL terms separately,
validation behaviour and runtime, and retain the best model chosen using only
normal validation images. Validation uses one latent sample per image, resetting
its random seed to 43 (training seed plus one) each epoch. The same noise sequence
is used for each epoch's validation pass; fitting RNG state is restored afterward.
This makes comparisons consistent within a run while retaining sampled decoding.

These settings are a starting point, not a claim of optimality. The implementation
uses PyTorch's [Dataset and DataLoader](https://docs.pytorch.org/docs/2.14/data.html)
for loading batches and [Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
for weight updates. Four CPU threads and `num_workers=0` keep this Windows run
straightforward. Seeds and software versions are recorded; identical results
across platforms or PyTorch versions are not guaranteed, as explained in the
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

### What the training code does

- `NormalBottleDataset` preprocesses originals at the chosen resolution and
  converts HWC to CHW; its default remains 128.
- `split_normal_images` shuffles sorted normal filenames with seed 42 and holds
  out 42 images. It writes no dataset files; the split is a list of filenames.
- `run_epoch` performs loss calculation and weight updates for fitting. With no
  optimizer supplied, it measures validation loss with gradients disabled.
- `load_saved_split` verifies and reuses E1's saved lists for E2 in their exact order.
- `training_configuration` describes the common settings and selected dimensions.
- `train_vae` saves the split, records each epoch, selects the best checkpoint by
  normal validation total loss and applies early stopping.
- `main` reads command options and starts a new run.

Losses are accumulated with each batch's image count, so the final small batch
is weighted correctly. Validation images never receive weight updates. No test
images or ground-truth masks are used. Images are not augmented or duplicated on
disk, and only saved results are written under `outputs/`.

Each run gets a new output directory. An existing run name is refused to protect
earlier results. Every invocation starts a fresh model; checkpoint resuming has
not been implemented.

### Actual initial training run

On 2026-10-06, E1 completed three epochs on the real dataset, using 167 fitting
images and 42 normal validation images. The saved split has zero overlap and
covers all 209 normal training images. The CPU run took approximately 35.6 seconds
and performed 63 optimizer steps. Early stopping did not trigger in three epochs.

| Epoch | Training total loss | Validation total loss | Validation reconstruction | Validation KL |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2703.242 | 982.892 | 850.517 | 132.375 |
| 2 | 436.683 | 215.696 | 184.313 | 31.384 |
| 3 | 163.448 | 127.805 | 109.014 | 18.791 |

Epoch 3 was the best observed normal-validation checkpoint. The loss decreased
during this short run. This is an initial training result, not a completed
50-epoch experiment or evidence of defect-detection performance. Subsequent full
training and normal-validation calibration are documented below; normal/defective
test evaluation remains outstanding.

The saved outputs in `outputs/E1/initial_training/` are:

- `data_split.json`: source folder, seed and exact training/validation filenames.
- `history.csv`: separate training and validation loss components for each epoch.
- `best_model.pt`: learned model weights, selected epoch, validation loss and settings.
- `training_summary.json`: settings, counts, optimizer steps, best epoch and elapsed time.

### PowerShell training commands

Inspect the completed initial run:

```powershell
Get-Content .\outputs\E1\initial_training\training_summary.json
Import-Csv .\outputs\E1\initial_training\history.csv | Format-Table
```

Run another short three-epoch training session in a new automatically named folder:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\train.py --epochs 3
```

The completed full run used this command, up to 50 epochs with patience 10:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\train.py --epochs 50 --patience 10 --run-name full_training
```

That run is saved already; choose a different `--run-name` to train again.
Settings may be supplied with
`--batch-size`, `--learning-rate` and `--seed`; defaults are 8, 0.001 and 42.
The same seed and unchanged filename list reproduce the same normal split.
All training outputs are excluded from Git. No temporary test scripts or
synthetic datasets were created for this step.

### Actual full E1 training run

On 2026-10-06, E1 completed all 50 requested epochs using the same 167 normal
fitting images and 42 normal validation images as the initial run, with zero
overlap. The CPU run took 508.3 seconds (8 minutes 28 seconds) and performed
1,050 optimizer steps. Early stopping did not trigger.

| Epoch | Training total loss | Validation total loss | Validation reconstruction | Validation KL |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2703.242 | 982.892 | 850.517 | 132.375 |
| 47 (selected) | 53.214 | 53.856 | 48.192 | 5.664 |
| 50 | 52.764 | 54.283 | 48.042 | 6.241 |

Epoch 47 had the lowest normal-validation **total** loss, so `best_model.pt`
contains that epoch's weights. The final epoch's reconstruction term alone was
lower, but checkpoint selection uses reconstruction plus KL. Outputs are in
`outputs/E1/full_training/`, with the same four training files described above.
The checkpoint loaded successfully. These normal-image losses do not establish
defect-detection performance.

## Phase 2.4: inspect training behaviour and reconstruction quality

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install "matplotlib>=3.9,<4"
.\.venv\Scripts\python.exe .\src\bottle_vae\inspect_training.py --run-folder .\outputs\E1\full_training
Get-Content .\outputs\E1\full_training\inspection\inspection_report.json
Invoke-Item .\outputs\E1\full_training\inspection\loss_curves.png
Invoke-Item .\outputs\E1\full_training\inspection\001_reconstruction.png
```

Matplotlib is added specifically for the loss plots, using its
[subplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
and [figure export](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
functions. The reusable inspection script:

- `load_history` reads the saved CSV values without smoothing or recalculating losses.
- `save_loss_curves` plots training and validation total, reconstruction and KL
  losses separately. Each y-axis uses a labelled logarithmic scale so the initial
  large losses do not hide later changes. A dashed line marks epoch 47. PNG and
  PDF versions are saved for viewing and use in the report.
- `save_reconstruction` combines the actual original, its prepared E1 input and
  the trained VAE's output into one labelled picture.
- `inspect_training` loads the checkpoint and selects the first, middle and last
  sorted filenames from the saved **validation** split, then saves figures and
  their provenance. `main` reads the run folder and dataset location.

The selected source files are exactly:

```text
data/bottle/train/good/001.png
data/bottle/train/good/087.png
data/bottle/train/good/206.png
```

Although these originals live in MVTec's `train/good` folder, they were held out
in our saved 42-image validation split and received no weight updates.
Preprocessing reads each actual original as RGB, resizes directly to 128 × 128
using bilinear resampling, and scales values to float32 `[0, 1]`. The model uses
`eval()`, inference mode and `sample=False`: it decodes the latent mean for
repeatable previews. The recorded training-validation losses used latent
sampling; these pictures use mean decoding and do not recalculate those losses.

Each comparison shows the 900 × 900 original, the 128 × 128 input, and the
128 × 128 reconstruction. For display only, all panels are shown at 320 × 320
with bilinear scaling. Model input and output values are rounded from `[0, 1]`
to `[0, 255]` for saving. There is no cropping, sharpening or individual contrast
adjustment. Source pictures come only from the real dataset; the reconstruction
panel is the trained VAE's prediction.

Generated figures and `inspection_report.json` are saved in
`outputs/E1/full_training/inspection/`, excluded from Git. The JSON records exact
absolute source paths, the checkpoint, selected epoch and display method. Rerunning
inspection replaces these derived figures; it does not change training files,
checkpoint weights or source images. No temporary scripts or mock data are used.

Inspection completed on 2026-10-06 using Matplotlib 3.11.2 and the epoch-47
checkpoint. All 50 epochs are plotted and all three selected real validation
images produced finite 128 × 128 RGB reconstructions. The plotted training and
validation losses both decrease and stay close through this run. In the previews,
the model preserves the overall circular bottle shape, but fine rim details are
blurrier than in the prepared input and the three outputs look very similar.
These are qualitative observations from three normal images, not a defect
detection result or a diagnosis of why detail is lost. Scoring and threshold
calibration are documented below; evaluation on the reserved test set remains
outstanding.

## Phase 3: anomaly scores and heatmaps

E1 image scoring, normal-validation threshold calibration and the requested
three normal-image heatmaps are now implemented. Reserved test-set performance
evaluation belongs to Phase 5, after the consistent configuration runs in Phase 4.

### Reconstruction-error score and normal-validation threshold

E1 now uses one **mean squared error (MSE)** score per image. Load and preprocess
the actual original in the same way as training: RGB, bilinear resize to
128 × 128, and float32 values divided by 255. Reconstruct it with the saved
epoch-47 checkpoint using the latent mean (`sample=False`). Then calculate:

```text
score = mean((input - reconstruction)^2) over all 128 * 128 * 3 RGB values
flag as anomaly when score > threshold
```

A larger score means a larger mismatch between input and reconstruction. The
score is calculated before preview resizing or conversion back to uint8. It
includes the entire image, including the background; a small local defect may
contribute relatively little to this average. Whether this baseline detects
defects must be measured on the reserved test set.

KL remains part of the training objective, but is not included in this
pixel-reconstruction score. For the **same** input and decoded image, E1's
half-summed reconstruction loss equals `24,576 * score`. The sampled validation
losses in `history.csv` use a different decoding convention and also report KL;
they are not the scores in `validation_scores.csv`.

### Threshold method chosen before test evaluation

The initial threshold is the **95th percentile** of reconstruction scores from
all 42 normal validation images listed in the saved `data_split.json`.
`numpy.percentile(scores, 95, method="linear")` sorts the scores and interpolates
between adjacent values when the percentile falls between them. This is NumPy's
documented [percentile method](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html).
Only errors strictly greater than the threshold are flagged; equality stays normal.

The 95th percentile is a recorded baseline choice, not a claim of optimality or
a guaranteed 5% false-positive rate on new normal images. With only 42 scores,
the number above the interpolated percentile need not be exactly 5%. The same
normal validation split was used for checkpoint selection, so its flagged
fraction is a calibration result, not an independent performance estimate.
Test images and labels were not consulted to choose this method or value.

### Actual E1 calibration result

On 2026-10-06, all 42 actual normal validation images were scored using the
epoch-47 checkpoint, with no weight updates. The calculation produced:

| Quantity | Actual result |
| --- | ---: |
| Normal validation images scored | 42 |
| Minimum score | 0.0013796249 |
| Mean score | 0.0020469056 |
| Maximum score | 0.0039538215 |
| 95th-percentile threshold | 0.0034759184 |
| Normal calibration images above threshold | 3 / 42 (7.14%) |

The three flagged files are genuine normal images from `data/bottle/train/good/`:

| Image | Reconstruction score |
| --- | ---: |
| `035.png` | 0.0034923286 |
| `055.png` | 0.0039538215 |
| `056.png` | 0.0035983229 |

These are false flags on the calibration data, not ground-truth defects.
The threshold lies between the fourth- and third-largest validation errors,
leaving three above it. A normal-only trained VAE still has non-zero errors on
normal images, and the decision rule can flag a normal image with a large error.
No normal or defective **test** image has been scored yet.

### Code, saved outputs and PowerShell commands

`scoring.py` uses the existing dependencies and small functions:

- `score_images` reads originals in batches, reconstructs using the latent mean
  with gradients disabled, and returns one pixel-averaged squared error per image.
- `choose_threshold` calculates the specified percentile of normal scores.
- `calibrate_threshold` reads the recorded validation split, confirms it is separate
  from fitting, loads the E1 checkpoint and saves the scores and threshold.
- `main` reads the run folder, dataset root, percentile and batch size.

Outputs are saved in `outputs/E1/full_training/calibration/`, excluded from Git:

- `validation_scores.csv`: exact source paths, each score and whether it exceeds
  the threshold. All source images belong to the normal validation split.
- `threshold.json`: the full-precision threshold, percentile method, score and
  decoding definitions, counts, flagged filenames and source locations. SHA-256
  hashes identify the exact checkpoint and saved split for later evaluation.

The command used to create these results was:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\scoring.py --run-folder .\outputs\E1\full_training --percentile 95
```

Calibration is recorded already. The script refuses an existing calibration
directory to protect the threshold from being overwritten. Inspect the saved
results with:

```powershell
Get-Content .\outputs\E1\full_training\calibration\threshold.json
Import-Csv .\outputs\E1\full_training\calibration\validation_scores.csv |
    Select-Object image, score, flagged_as_anomaly | Format-Table
```

No temporary scripts, mock data or new dependencies were used. Dataset images,
masks and checkpoint weights are read-only throughout calibration. Heatmaps
complete the outstanding E1 work in Phase 3, as documented below. Phase 4 comes
next; reserved normal and defective test evaluation is deferred to Phase 5.

### E1 spatial reconstruction-error heatmaps

`heatmaps.py` reuses `preprocess_image` and the E1 model. It takes its source paths
from the existing `outputs/E1/full_training/inspection/inspection_report.json`
and restricts image inputs to the same three held-out normal originals:

```text
data/bottle/train/good/001.png
data/bottle/train/good/087.png
data/bottle/train/good/206.png
```

The code confirms that these files belong to the saved validation split and are
absent from fitting. It loads the selected epoch-47 checkpoint, checks its hash
against the existing calibration record, and reconstructs using the latent mean
(`sample=False`) with gradients disabled. It reads no reserved test images or
masks and performs no training or threshold recalibration.

For each prepared float32 RGB input and reconstruction, calculate:

```text
squared RGB differences = (input - reconstruction)^2
error map[y, x]         = mean(squared RGB differences[y, x, :])
image score             = mean(error map)
```

The map is a **128 × 128 float32 array**. Averaging the three colour-channel
errors at each pixel preserves their spatial locations. Averaging all map pixels
then gives the same MSE as averaging over all RGB values in the image. Errors are
computed from the prepared arrays, before any display enlargement or rounding.

The code checks each map mean against both the directly computed RGB image MSE
and that image's existing score in `calibration/validation_scores.csv`.
It uses `numpy.isclose` with `rtol=1e-5` and `atol=1e-8`, following its documented
[tolerance rule](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html).
Small differences can result from float32 reduction order or batch calculations.
All six comparisons passed in the actual run on 2026-10-06; the largest absolute
difference was approximately **2.33 × 10⁻¹⁰**.

| Source image | Mean spatial error | Matches RGB MSE and saved score within tolerance |
| --- | ---: | --- |
| `001.png` | 0.0020226978 | Yes |
| `087.png` | 0.0019886440 | Yes |
| `206.png` | 0.0015118895 | Yes |

Each labelled PNG shows the prepared input, its latent-mean reconstruction and
the coloured spatial error map, with a numerical colour bar. All three use the
same **linear `magma` colour scale**, from **0 to 0.1220361963**. The upper bound
is the largest spatial error across the three maps, calculated once and reused
for every figure. Each colour therefore represents the same error value across
images; maps are not individually contrast-normalised. This scale is specific
to these three figures, not a universal scale for future configurations.

All panels use their native 128 × 128 arrays and are enlarged to fill the figure
panels with Matplotlib's nearest-neighbour display interpolation. The PNGs are
3600 × 1440 at 300 dpi; the enlarged panels repeat native pixels and restore no
detail. Calculations remain at 128 × 128. The display method is explicit in
[Matplotlib's image API](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html).

**These heatmaps show reconstruction differences, not defect probabilities or
confirmed defect locations.** The brighter rim regions show where input and
reconstruction differ in these normal images. The image-level anomaly threshold
is not used as a pixel-level cutoff, and no defect masks or pixel classifications
are produced.

### Heatmap code, outputs and PowerShell commands

The new file has small functions:

- `load_inspection_sources` confirms the three recorded original paths and their
  held-out status.
- `calculate_error_maps` computes squared differences, spatial channel averages
  and whole-image RGB scores.
- `save_heatmap` saves the three labelled panels with the shared scale and colour bar.
- `make_heatmaps` loads the checkpoint, performs the score checks and saves figures
  and provenance. `main` reads the run folder.

Outputs are saved in `outputs/E1/full_training/heatmaps/`, excluded from Git:

- `001_heatmap.png`, `087_heatmap.png`, `206_heatmap.png`: labelled comparison figures.
- `heatmap_report.json`: exact source paths and hashes, inspection report and
  checkpoint provenance, map definition, shared colour scale, numerical scores,
  tolerance checks and display method.

To inspect the completed outputs from the repository root:

```powershell
Invoke-Item .\outputs\E1\full_training\heatmaps
Invoke-Item .\outputs\E1\full_training\heatmaps\001_heatmap.png
Get-Content .\outputs\E1\full_training\heatmaps\heatmap_report.json
```

To regenerate only these derived heatmap outputs:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\heatmaps.py --run-folder .\outputs\E1\full_training
```

Regeneration replaces only the derived heatmap figures and report. It uses the
existing dependencies and creates no temporary scripts or synthetic inputs.
Original image, checkpoint and threshold file hashes were unchanged after the
completed run. Verification helpers remain in place. The requested E1 Phase 3
work is complete; adapting the pipeline and completing E2/E3 belongs to Phase 4,
before test-set performance evaluation in Phase 5.

## Phase 4: E2 preparation and training

E2 preparation, implementation, training, normal-image inspection, calibration
and heatmaps are complete. The run completed 50 epochs and selected the epoch-45
checkpoint, which is also used for all inspection and calibration outputs.
All preparation, fitting and validation inputs came from the actual `data/bottle/train/good/`
dataset. No reserved test images, masks, synthetic data or temporary test
scripts were used for E2. E3 preparation is documented separately below. E1's saved results, checkpoint and
threshold remain unchanged; verification helpers remain in place.

### Configuration and architecture

| Setting | E2 value |
| --- | --- |
| Input / reconstructed image | 256 × 256 RGB, float32 in `[0, 1]` |
| Preprocessing | Resize directly from 900 × 900 originals with bilinear resampling; divide by 255 |
| Latent dimension | 32 |
| Convolution blocks | Same four blocks and channels as E1; kernel 4, stride 2, padding 1 |
| Activations | ReLU; final RGB sigmoid |
| Fitting / validation | E1's exact saved 167 / 42 filename lists, in the same order |
| Optimizer / learning rate / batch size | Adam / 0.001 / 8 |
| Seeds | Initialization and fitting shuffle: 42; sampled validation: 43 reset each epoch |
| Execution | CPU, four threads, DataLoader workers 0 |
| Budget / early stopping | At most 50 epochs; stop after 10 epochs without improvement |
| Checkpoint selection | Strictly lowest normal validation total loss `R + K` |
| Training objective | Same half-summed RGB reconstruction error plus summed latent KL |
| Additional reporting | Pixel-averaged RGB reconstruction MSE and KL per latent dimension |
| Implemented parameter count | 3,524,515 |

`BottleVAE(image_size=256, latent_dim=32)` changes the spatial sizes and the
three linear layers around the latent representation. The convolutions retain
their original settings. The four encoder blocks produce:

```text
(B, 3, 256, 256)
 -> (B, 16, 128, 128)
 -> (B, 32, 64, 64)
 -> (B, 64, 32, 32)
 -> (B, 128, 16, 16)
 -> flatten to (B, 32768)
 -> separate mean and log-variance heads, each (B, 32)
```

The decoder maps 32 latent values to 32,768 features and reshapes to
`(B, 128, 16, 16)`. The same four transposed convolution blocks then produce
`(B, 64, 32, 32)`, `(B, 32, 64, 64)`, `(B, 16, 128, 128)` and finally
`(B, 3, 256, 256)`. Training and validation explicitly sample the latent
distribution. Evaluation-mode reconstruction uses the latent mean by default,
as in E1. E2 starts with fresh weights; E1's checkpoint is retained for E1.

### Completed preparation checks

Verified on 2026-10-06:

| Check | Actual result |
| --- | --- |
| Normal originals readable | 209 / 209; all 900 × 900 RGB |
| Prepared images | 209 / 209 |
| Prepared shape / type | `(256, 256, 3)` / float32 for every image |
| Finite values and range check | Passed for every image; all values within `[0, 1]` |
| Observed prepared range | 0.11372549086809158 to 1.0 |
| Preprocessing failures | None |
| Split membership / overlap | Exactly the 209 actual normal PNGs; 167 fitting, 42 validation, zero overlap |
| Split filename order | Preserved for both lists and dataset objects |
| Copied split manifest hash | Matches the E1 source manifest byte for byte |
| Model check sources | `data/bottle/train/good/000.png`, `104.png`, `208.png` |
| Model batch input / output | Both `(3, 3, 256, 256)`; single-image output `(1, 3, 256, 256)` |
| Encoder features / latent heads | `(3, 128, 16, 16)` / two `(3, 32)` tensors |
| Model tensor type / values | All float32 and finite; reconstructed RGB values within `[0, 1]` |
| Reporting verification | R+K, MSE and KL per dimension match direct calculations within tolerance |
| Weight updates during preparation checks | Zero; model parameters unchanged by those checks |
| E1 compatibility | Epoch-47 checkpoint loads strictly; defaults remain 128/16 with 747,395 parameters |
| Shared training settings | Match E1's saved checkpoint configuration |

The split source is `outputs/E1/full_training/data_split.json`. Its SHA-256 is:

```text
1e72751ea7c3324521e40a42a6a7e838df031c367e3ded11a16788e765120b49
```

The filename lists are loaded directly; they are not regenerated from the seed.
Their membership, uniqueness, counts and zero overlap are checked against the
actual normal source folder. Fitting is shuffled by the DataLoader during
training, using seed 42; validation retains the saved order. The model check
also verifies the new reporting calculations with batches of two and one actual
images, ensuring the final smaller batch is weighted correctly. It uses relative
tolerance `1e-6` and absolute tolerance `1e-8`, with no optimizer or gradients.
These untrained-model checks establish implementation behaviour, not
reconstruction quality or defect-detection performance.

### Reports and commands

All E2 preparation files are under `outputs/E2/`, excluded from Git:

- `preprocessing_report.json`: all 209 original and prepared-image checks.
- `model_check_report.json`: actual source paths, model shapes, parameter count,
  reporting checks, split provenance and E1 compatibility.
- `training_configuration.json`: proposed settings from the same function used
  by the training command; this file records settings and is not a CLI input file.
- `data_split.json`: an exact byte copy of E1's saved split manifest.
- `split_verification.json`: membership, order, overlap and source/copy SHA-256 checks.

Inspect or repeat these preparation checks from the repository root:

```powershell
Get-Content .\outputs\E2\preprocessing_report.json
Get-Content .\outputs\E2\model_check_report.json
Get-Content .\outputs\E2\split_verification.json
.\.venv\Scripts\python.exe .\src\bottle_vae\preprocessing.py --experiment E2
.\.venv\Scripts\python.exe .\src\bottle_vae\model_check.py --experiment E2 --split-file .\outputs\E1\full_training\data_split.json
```

These check commands replace E2's derived preparation reports only; they do not
create a training run. Prepared arrays are generated on demand in memory rather
than saved as another dataset.

The following command was used for the completed E2 run:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\train.py --experiment E2 --split-file .\outputs\E1\full_training\data_split.json --epochs 50 --batch-size 8 --learning-rate 0.001 --patience 10 --seed 42 --run-name full_training
```

It created the new `outputs/E2/full_training/` directory containing
`data_split.json`, `history.csv`, `best_model.pt` and `training_summary.json`.
Existing run directories are refused with `exist_ok=False`; this completed run
cannot be overwritten by repeating the command. E1 defaults and `outputs/E1/`
remain separate.

For new runs, `history.csv` records fitting and validation total, reconstruction
and KL losses plus `train_reconstruction_mse`, `validation_reconstruction_mse`,
`train_kl_per_dimension` and `validation_kl_per_dimension`. Each epoch's
`seconds` measures its fitting and validation passes. `training_summary.json`
records the implemented parameter count, source split hash, epochs, optimizer
steps, selected checkpoint, UTC timestamps and total elapsed seconds. Total
timing follows E1's convention: start after model/data-loader initialization and
split writing, include epoch work, history and checkpoint writes, and exclude
the final summary write. The actual E2 elapsed time is recorded below.

### Actual E2 training results

On 2026-10-06, E2 completed all **50 requested epochs**, with **1,050 optimizer
steps**. Early stopping did not occur. The checkpoint selected by the lowest
normal validation total loss is **epoch 45**, saved as
`outputs/E2/full_training/best_model.pt`. The objective remained `R + K` with
no change to its reductions or weighting, and the run used the verified settings
listed above.

Metrics from the selected epoch's `history.csv` row:

| Quantity | Fitting | Normal validation |
| --- | ---: | ---: |
| Total loss, R + K | 223.31878689520374 | 229.59783499581474 |
| Reconstruction loss, R | 205.79910278320312 | 211.65382167271204 |
| KL loss, K | 17.519684763011817 | 17.944012868972052 |
| Pixel-averaged RGB reconstruction MSE | 0.0020934967303224755 | 0.0021530540238711097 |
| KL per latent dimension | 0.5474901488441193 | 0.5607504021553766 |

These are sampled-latent training/validation quantities, averaged over images.
Fitting values aggregate batches while weights are updated; validation is
measured after that epoch, without updates, using seed 43. They are not the
latent-mean reconstruction scores that will be used for later calibration.
The final epoch-50 validation total was 230.80099196661087, so the selected
epoch-45 checkpoint was retained.

| Execution detail | Actual result |
| --- | --- |
| Completion status | `completed_requested_epochs`; process exit code 0 |
| Total recorded training time | 864.4417711000424 seconds, approximately 14 minutes 24 seconds |
| CPU | AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx; 4 physical cores, 8 logical processors |
| Execution threads / DataLoader workers | 4 CPU threads / 0 workers; CPU training, no GPU |
| PyTorch | `2.14.1+cpu` |
| Total trainable model parameters | 3,524,515 |
| Failures / non-finite losses | None reported; all 50 history rows contain finite metrics |
| Fitting / validation split | Exact E1 167 / 42 lists, original order, zero overlap |

The selected epoch matches the minimum validation total in all 50 recorded
epochs. Its loss agrees with the training summary within numerical tolerance.
The run's copied split manifest has the same SHA-256 as E1's source manifest
documented above. Before/after hash checks confirmed all **22 E1 artifacts** and
all **209 original normal training images** were unchanged. No existing run
directory was overwritten. Only the new E2 training outputs and this README were
updated for this training step; no pipeline code or settings were changed.

Inspect the saved results without rerunning training:

```powershell
Get-Content .\outputs\E2\full_training\training_summary.json
Import-Csv .\outputs\E2\full_training\history.csv | Where-Object { $_.epoch -eq '45' }
Invoke-Item .\outputs\E2\full_training
```

No ranking of E1 and E2 is made from raw total loss: their pixel counts and latent
dimensions differ, as described in the loss-scaling section. Normal-image
reconstruction quality is inspected below; defect-detection performance has
not been evaluated.

### E2 normal-image inspection, calibration and heatmaps

Completed on 2026-10-07 using the existing helpers, with dimensions and figure
labels now read from the checkpoint configuration. Their commands still default
to E1; passing `--run-folder .\outputs\E2\full_training` selects E2's saved run.
No training, tuning, temporary scripts or synthetic inputs were used.

The unchanged epoch-45 checkpoint has SHA-256:

```text
4c653c94f62e20b9b97ae52b3cb9ea56b03accef352a48f59ddde75906457278
```

#### Loss curves and reconstruction inspection

`inspection/loss_curves.png` and `.pdf` plot all 50 saved epochs' fitting and
validation total, reconstruction and KL losses, with logarithmic vertical axes
and a dashed line at epoch 45. Values are taken directly from `history.csv`,
without smoothing or recalculating training losses. The curves drop sharply
over the first few epochs, then improve more slowly with fluctuations. Fitting
and validation curves stay relatively close, without a sustained widening gap
in these plots. This describes behaviour on the normal fitting/validation data,
not independent generalisation or defect-detection performance.

E2 reads the exact three sources from E1's saved inspection report:

```text
data/bottle/train/good/001.png
data/bottle/train/good/087.png
data/bottle/train/good/206.png
```

All three are confirmed present in E2's saved validation list and absent from
fitting. The selected checkpoint is loaded strictly as the 256/32 model. The
model decodes the latent mean (`sample=False`) with gradients disabled and no
weight updates. Its reconstruction batch is `(3, 3, 256, 256)`, with finite RGB
values between 0.12372057139873505 and 1.0.

Each labelled comparison shows the 900 × 900 original, prepared 256 × 256 input
and 256 × 256 reconstruction. All three panels are displayed at 320 × 320 with
bilinear scaling: the original is reduced and the prepared/reconstructed views
are enlarged by a factor of 1.25 per side. There are no crops, sharpening or
per-image contrast adjustments. The reconstructions reproduce the broad bottle
outline, dark centre and light background, but blur fine rim textures and
reflections. The three reconstructions also look more similar than the inputs.
These normal details contribute reconstruction error without being defects.
Successful reconstruction of these images does not establish detection of
unseen defects, particularly when small features may be smoothed.

`inspection/inspection_report.json` records the exact original paths and hashes,
validation membership, E1 source-report provenance, checkpoint and history
hashes, dimensions, decoding and display methods, and figure locations.

#### E2's own normal-validation threshold

All **42** saved normal validation images were scored with latent-mean decoding,
in sorted filename order, using batches of eight on the CPU with four threads.
The score is the mean squared difference over all `256 * 256 * 3` prepared and
reconstructed RGB values. It contains no KL term or half-sum multiplier.
All scores were finite, with no missing, duplicate or unexpected validation
filenames and no fitting overlap.

| Calibration quantity | Actual E2 result |
| --- | ---: |
| Minimum normal score | 0.0013798618456348777 |
| Mean normal score | 0.0022004102827936764 |
| Maximum normal score | 0.0040403385646641254 |
| 95th-percentile threshold, full precision | **0.0033652833313681184** |
| Normal calibration images flagged | **3 / 42 (7.14%)** |

The threshold is calculated from E2's own scores using
`numpy.percentile(scores, 95, method="linear")`, exactly the percentile method
used for E1. The decision rule remains `score > threshold`; equality stays
normal. E1's numerical threshold is not reused. The flagged normal filenames
are `035.png`, `055.png` and `056.png`. A finite-sample 95th percentile does not
require exactly 5% of images to be flagged.

These are **normal calibration flags, not reserved test results or confirmed
defects**. The same validation set was used for checkpoint selection and
calibration, so its flagged fraction is not an independent estimate of the
false-positive rate on new images. Neither the percentile nor settings were
tuned after viewing these results. The threshold is recorded for later Phase 5
evaluation; the helper refuses to overwrite an existing calibration directory.

`calibration/validation_scores.csv` saves every original path, full-precision
score and flag. `calibration/threshold.json` records the threshold, method,
decision rule, score summaries, flagged names, versions and checkpoint/split
hashes. These mean-decoded scores need not equal the sampled-latent validation
MSE recorded during training.

#### Spatial error maps and display scale

The three inspection images have float32 **256 × 256** spatial error maps:

```text
squared RGB differences = (prepared input - latent-mean reconstruction)^2
error map[y, x]         = mean(squared RGB differences[y, x, :])
image MSE              = mean(error map)
```

| Normal image | Mean spatial error | Matches direct RGB MSE and saved calibration score |
| --- | ---: | --- |
| `001.png` | 0.0019772674422711134 | Yes |
| `087.png` | 0.0019881653133779764 | Yes |
| `206.png` | 0.0013798619620501995 | Yes |

All six equivalence checks passed using `rtol=1e-5` and `atol=1e-8`. The largest
absolute difference was `1.1641532182693481e-10`, from float32 reduction order.
All three inspected image scores are below the recorded E2 threshold.

All three figures share one **linear `magma` scale from 0 to
0.24009953439235687**, with a numerical colour bar. Its upper bound is the
largest spatial error across these three E2 maps, computed once and reused;
there is no separate contrast normalisation per map. E1 and E2 have different
upper bounds, so this scale is shared within the three E2 figures.

Each figure shows the prepared input, reconstruction and coloured error map.
The native 256 × 256 arrays are enlarged to fill the figure panels using
nearest-neighbour interpolation; the PNG canvas is 3600 × 1440 at 300 dpi.
Enlargement repeats native pixels and adds no detail. Error calculations use
the native float32 arrays before display scaling or rounding.

Visible errors concentrate around normal rim details and reflections that the
reconstruction smooths. **Heatmaps show reconstruction differences, not defect
probabilities or confirmed defect locations.** The image-level threshold is not
used as a pixel-level cutoff, and no masks or pixel classifications are made.
`heatmaps/heatmap_report.json` records original-image hashes, checkpoint/split/
inspection/calibration provenance, numerical score checks, scale and display
details.

#### Saved outputs and PowerShell commands

All new outputs are under `outputs/E2/full_training/`, excluded from Git:

- `inspection/`: `loss_curves.png`, `loss_curves.pdf`,
  `001_reconstruction.png`, `087_reconstruction.png`, `206_reconstruction.png`
  and `inspection_report.json`.
- `calibration/`: `validation_scores.csv` and `threshold.json`.
- `heatmaps/`: `001_heatmap.png`, `087_heatmap.png`, `206_heatmap.png` and
  `heatmap_report.json`.

The commands used, in order, were:

```powershell
.\.venv\Scripts\python.exe -B .\src\bottle_vae\inspect_training.py --run-folder .\outputs\E2\full_training
.\.venv\Scripts\python.exe -B .\src\bottle_vae\scoring.py --run-folder .\outputs\E2\full_training --percentile 95 --batch-size 8
.\.venv\Scripts\python.exe -B .\src\bottle_vae\heatmaps.py --run-folder .\outputs\E2\full_training
```

Calibration is already recorded; its command refuses an existing directory.
The inspection and heatmap commands can regenerate their derived figures.
Inspect the completed outputs without recalculating them:

```powershell
Invoke-Item .\outputs\E2\full_training\inspection
Invoke-Item .\outputs\E2\full_training\heatmaps
Get-Content .\outputs\E2\full_training\calibration\threshold.json
Get-Content .\outputs\E2\full_training\heatmaps\heatmap_report.json
```

Before/after hash checks confirmed all 22 E1 artifacts, all four existing E2
training files (including the epoch-45 checkpoint), and all 209 original normal
images were unchanged. No reserved test images or masks were accessed, no new
dependencies were added, and verification helpers remain.

**E2's planned work before test evaluation is complete.** E3's subsequent training,
normal-image inspection, calibration and heatmaps are documented below and
complete the agreed Phase 4 work. Reserved test-set performance evaluation remains
Phase 5; the paper and demo remain Phase 6. These normal-image outputs do not
answer the research question about detecting unseen defects.

## Phase 4: E3 preparation, implementation and training

E3 preparation and implementation were verified on actual normal dataset images
on 2026-10-07. That preparation did not create a training run or checkpoint.
The subsequent training run completed on 2026-10-09 and is documented below.
Preprocessing already supported 512 × 512, so it needed no code changes. The
existing model, training configuration and verification helper now accept E3;
the downstream inspection, scoring and heatmap configuration guards also accept
E3 for use after a trained checkpoint exists. Those downstream helpers were not
run for E3 during preparation.

### Verified configuration

| Setting | E3 value |
| --- | --- |
| Prepared input / reconstructed output | 512 × 512 RGB, float32 |
| Latent dimension | 64 |
| Architecture | Same four encoder and decoder convolution blocks as E1/E2 |
| Channels | Encoder 3 → 16 → 32 → 64 → 128; decoder 128 → 64 → 32 → 16 → 3 |
| Kernels / strides / padding | 4 / 2 / 1 throughout |
| Activations | ReLU; final RGB sigmoid |
| Optimizer / learning rate | Adam / 0.001 |
| Batch size | 8, retained without adjustment |
| Seeds | Initialization/fitting shuffle 42; sampled validation 43 reset each epoch |
| Execution | CPU, four threads, DataLoader workers 0; PyTorch `2.14.1+cpu` |
| Maximum epochs / patience | 50 / 10 |
| Checkpoint selection | Strictly lowest normal validation total `R + K` |
| Objective | Existing half-summed squared RGB error plus summed latent KL |
| Reporting only | Pixel-averaged reconstruction MSE and KL per latent dimension |
| Trainable parameters, including biases | **25,642,979** |

`BottleVAE(image_size=512, latent_dim=64)` retains the existing encoder/decoder
layers and computes their feature sizes from the resolution:

```text
(B, 3, 512, 512)
 -> (B, 16, 256, 256)
 -> (B, 32, 128, 128)
 -> (B, 64, 64, 64)
 -> (B, 128, 32, 32)
 -> flatten to (B, 131072)
 -> separate mean and log-variance heads, each (B, 64)
```

The decoder expands 64 latent values to 131,072 features, reshapes to
`(B, 128, 32, 32)`, and uses the same four transposed convolutions to return
`(B, 3, 512, 512)`. E1's constructor and dataset-loader defaults remain 128/16
and 128 respectively; saved E1/E2 state-dictionary names and shapes are preserved.

### Why the parameter count grows

The flattened representation is 8,192 values for E1, 32,768 for E2 and 131,072
for E3. Relative to E2, E3 has four times as many flattened features and twice
as many latent values. Each large linear weight matrix therefore contains eight
times as many weights. The number of convolution parameters stays unchanged.
The implemented E3 counts were measured from `model.parameters()`:

| Parameter group | Actual count |
| --- | ---: |
| Encoder convolutions | 173,040 |
| Mean head, 131,072 → 64 | 8,388,672 |
| Log-variance head, 131,072 → 64 | 8,388,672 |
| Decoder input, 64 → 131,072 | 8,519,680 |
| Decoder convolutions | 172,915 |
| **Total** | **25,642,979** |

The three linear layers account for 25,297,024 parameters; the convolutions
account for 345,955. The 64-dimensional latent representation is distinct from
this total number of learned weights and biases.

E3 has `512 * 512 * 3 = 786,432` RGB values per image. With the unchanged loss,
`R = 393,216 * pixel-averaged MSE`, and K sums over 64 latent dimensions.
Reporting records `2*R/786432` and `K/64`, without changing gradients, checkpoint
selection or early stopping. At equal pixel-averaged MSE and equal KL per latent
dimension, E3's R is four times E2's and K is twice E2's; their relative weighting
therefore changes. This is part of the agreed comparison of whole configurations,
not a reason to rank them by raw total loss.

### Completed checks on 2026-10-07

| Check | Actual result |
| --- | --- |
| Original normal images read | 209 / 209; all 900 × 900 RGB |
| Images prepared successfully | 209 / 209 |
| Prepared shape / dtype | `(512, 512, 3)` / float32 for every image |
| Finiteness / range | All finite and within `[0, 1]` |
| Observed range across all prepared images | 0.10196078568696976 to 1.0 |
| Preprocessing failures | None |
| Core architecture sources | `data/bottle/train/good/000.png`, `104.png`, `208.png` |
| Core batch input / output | `(3, 3, 512, 512)`; single-image output `(1, 3, 512, 512)` |
| Encoder features / latent heads | `(3, 128, 32, 32)` / two `(3, 64)` tensors |
| Reporting calculations | R+K, pixel-averaged MSE and KL per dimension match direct calculations |
| Full batch-size-8 input / output | Both `(8, 3, 512, 512)`; latent heads `(8, 64)` |
| Full batch-size-8 forward/loss check | Passed with finite tensors and losses; no size reduction |
| Optimizer steps / gradients | Zero optimizer steps; no gradients created |
| E1 checkpoint compatibility | Epoch 47 loads strictly; 128/16 dimensions and 747,395 parameters retained |
| E2 checkpoint compatibility | Epoch 45 loads strictly; 256/32 dimensions and 3,524,515 parameters retained |
| Shared training settings | Match both saved E1/E2 checkpoint configurations |
| Model-check failures | None |

The full batch uses the first eight fitting filenames in the unchanged saved
order: `037.png`, `204.png`, `100.png`, `085.png`, `195.png`, `167.png`, `064.png`
and `201.png`. Its check loads actual images, performs a sampled forward pass
and calculates the existing loss with gradients disabled. **This preparation
check did not test backward-pass or Adam-state memory**, so it verifies forward/loss execution at
batch size 8, not the memory required by a complete training step. The batch
size remains 8; a memory failure must be reported rather than silently reducing
it. Untrained loss values verify the calculations, not reconstruction quality
or defect-detection performance.

### Exact split reuse and separate reports

E3 directly loads `outputs/E1/full_training/data_split.json`; it does not make a
new random split. The checks confirmed exactly 167 fitting and 42 validation
filenames, uniqueness, zero overlap, complete membership of the 209 actual normal
PNGs and unchanged filename order in both lists and dataset objects. E3's copy
of the manifest is byte-identical to the source, with the same SHA-256:

```text
1e72751ea7c3324521e40a42a6a7e838df031c367e3ded11a16788e765120b49
```

Outputs are saved under `outputs/E3/`, excluded from Git:

- `preprocessing_report.json`: checks and failures for all 209 originals and
  their prepared arrays.
- `model_check_report.json`: actual source paths, tensor dimensions, parameter
  groups, reporting checks, full batch check and saved checkpoint compatibility.
- `training_configuration.json`: agreed settings from the training code's
  configuration function; a record of settings, not a CLI input file.
- `data_split.json`: exact E1 source manifest copy.
- `split_verification.json`: membership/order checks and source/copy hashes.

The preparation commands used were:

```powershell
.\.venv\Scripts\python.exe -B .\src\bottle_vae\preprocessing.py --experiment E3
.\.venv\Scripts\python.exe -B .\src\bottle_vae\model_check.py --experiment E3 --split-file .\outputs\E1\full_training\data_split.json
```

They save derived E3 reports only, without training or storing another image
dataset. Original images remain read-only. No synthetic data, temporary scripts,
reserved test images or masks were used, and verification helpers remain.
Before/after SHA-256 checks confirmed that all 43 E1/E2 output files and all
209 original normal training images were unchanged.

The following command was used for the completed E3 training run:

```powershell
.\.venv\Scripts\python.exe .\src\bottle_vae\train.py --experiment E3 --split-file .\outputs\E1\full_training\data_split.json --epochs 50 --batch-size 8 --learning-rate 0.001 --patience 10 --seed 42 --run-name full_training
```

It created `outputs/E3/full_training/` and recorded the same history, best
checkpoint, parameter count and runtime outputs as E1/E2. The existing
`mkdir(..., exist_ok=False)` refusal to overwrite run directories remains.
Repeating this command will refuse the existing run directory.

### Actual E3 training results

On 2026-10-09, E3 completed **23 epochs** and **483 optimizer steps**.
Early stopping occurred after ten consecutive epochs without an improvement
in normal validation total loss (epochs 14 through 23). The selected checkpoint
is **epoch 13**, saved as `outputs/E3/full_training/best_model.pt`.
The run retained 512 × 512 RGB inputs, 64 latent dimensions, batch size 8 and
all settings in the verified configuration above. No pipeline code or training
settings were changed; gradients used the existing `R + K` objective.

Metrics from the selected epoch's `history.csv` row:

| Quantity | Fitting | Normal validation |
| --- | ---: | ---: |
| Total loss, R + K | 996.5108708364521 | 991.1660505022321 |
| Reconstruction loss, R | 937.6668905840662 | 934.8700590587798 |
| KL loss, K | 58.843978082348485 | 56.29600125267392 |
| Pixel-averaged RGB reconstruction MSE | 0.0023846102050299237 | 0.0023774974868588502 |
| KL per latent dimension | 0.9194371575366951 | 0.87962501957303 |

These are sampled-latent fitting/validation quantities averaged over images.
Fitting values aggregate batches during weight updates; validation is measured
after the epoch without updates, using seed 43. They are not latent-mean
reconstruction scores or threshold-calibration results. The final epoch-23
validation total was 991.236822219122, so epoch 13 remained selected.

| Execution detail | Actual result |
| --- | --- |
| Completion status | `early_stopped`; process exit code 0 |
| Epochs / optimizer steps | 23 / 483 |
| Total recorded training time | 1605.679882899858 seconds, approximately 26 minutes 46 seconds |
| CPU | AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx; 4 physical cores, 8 logical processors |
| Execution threads / DataLoader workers | 4 CPU threads / 0 workers; CPU training, no GPU |
| PyTorch | `2.14.1+cpu` |
| Total trainable model parameters | 25,642,979 |
| Failures / non-finite losses | None reported; all 23 history rows contain finite metrics |
| Fitting / validation split | Exact E1 167 / 42 filename lists, original order, zero overlap |

Recorded timing uses the same convention as E1/E2: it starts after model and
data-loader initialization and split writing, includes epoch work, history and
checkpoint writes, and excludes the final summary write. Training successfully
performed backward passes and Adam updates at the unchanged batch size 8.

The selected epoch matches the minimum validation total in the 23 saved history
rows and agrees with the training summary. The run's copied split manifest has
the same SHA-256 as E1's source manifest documented above. Before/after hash
checks confirmed all **43 E1/E2 artifacts** and all **209 original normal training
images** were unchanged. No existing run directory was overwritten.

The new files under `outputs/E3/full_training/`, excluded from Git, are:

- `data_split.json`: byte-identical E1 fitting/validation manifest.
- `history.csv`: all 23 epochs' fitting and validation losses, normalized
  reporting metrics and epoch durations.
- `best_model.pt`: selected epoch-13 model weights and configuration.
- `training_summary.json`: settings, split verification, parameter count,
  epochs, optimizer steps, selected checkpoint, status and total runtime.

Inspect the saved results without rerunning training:

```powershell
Get-Content .\outputs\E3\full_training\training_summary.json
Import-Csv .\outputs\E3\full_training\history.csv | Where-Object { $_.epoch -eq '13' }
Invoke-Item .\outputs\E3\full_training
```

Only the new E3 training outputs and this README were updated for this training
step. No synthetic data or temporary scripts were used. Reserved test images
and masks were not accessed, and verification helpers remain.
No configuration ranking is made from raw total loss because resolutions and
latent dimensions differ. Defect-detection performance has not been evaluated.

E3's subsequent normal-image inspection, calibration and heatmaps are documented
below. Reserved test evaluation stays in Phase 5.

### E3 normal-image inspection, calibration and heatmaps

Completed on 2026-10-09 using the existing helpers without code changes or
retraining. Each helper strictly loaded the unchanged epoch-13 checkpoint as
the 512 × 512 RGB / latent-64 model. Reconstructions use the latent mean
(`sample=False`) with gradients disabled and no weight updates. Its SHA-256 is:

```text
1c1a342d07e5bc0ff0ed19c3ac919c8917244a1fffcc749ff4b68f74399699e5
```

#### Saved losses and reconstruction observations

`inspection/loss_curves.png` and `.pdf` show all **23 saved epochs**, with fitting
and normal-validation total, reconstruction and KL losses on logarithmic axes.
A dashed line marks selected epoch 13. The values are read directly from
`history.csv`, without smoothing or recomputing training losses. Total and
reconstruction losses drop sharply during the early epochs and then fluctuate
around a plateau. The fitting and validation curves remain relatively close in
these plots. KL falls sharply at first and subsequently fluctuates; it does not
decrease monotonically.

Early stopping ended training at epoch 23 because epochs 14–23 did not improve
the lowest validation **total R + K**, recorded at epoch 13. The last total,
991.236822219122, remained above the selected 991.1660505022321. The plot ends
at epoch 23 because the remaining maximum budget of 50 epochs was not used.
This describes normal-image training behaviour, not defect-detection performance.

Source paths are taken from E1's saved inspection report and match E2's report:

```text
data/bottle/train/good/001.png
data/bottle/train/good/087.png
data/bottle/train/good/206.png
```

All three are in E3's saved validation list and absent from fitting. Exact
absolute paths, source hashes and membership checks are recorded in
`inspection/inspection_report.json`. The reconstructed batch is
`(3, 3, 512, 512)`, with finite output values ranging from 0.12107117474079132
to 1.0.

The labelled comparisons show the original 900 × 900 image, prepared 512 × 512
input and latent-mean reconstruction. Each complete panel is displayed at
320 × 320 with bilinear scaling: this reduces the 512 × 512 arrays for display
only. Fixed `[0, 1]` to uint8 conversion is used, with no crops, sharpening or
per-image contrast adjustment. Scoring uses float32 arrays before this display
conversion.

Visual inspection shows that the reconstructions capture the circular bottle
outline, broad concentric rings and pale background. Fine orange reflections,
irregular rim texture and small highlights are blurred or missing. The three
reconstructions look very similar despite visible differences in their inputs.
These are reconstruction limitations on normal images; they do not establish
how the model will respond to defects.

#### E3's own normal-validation threshold

All **42 normal validation images** were scored in batches of eight using
latent-mean decoding and mean squared error over all `512 * 512 * 3` RGB values.
KL is excluded from this image score. The saved score filenames exactly match
the validation membership, with no fitting overlap, and all scores are finite.

The existing helper calculated `numpy.percentile(scores, 95, method="linear")`.
The unchanged decision rule is **score > threshold**; equality stays normal.
The threshold was calculated from E3's own scores, without reusing E1/E2 values.

| Calibration quantity | Actual value |
| --- | ---: |
| 95th-percentile threshold | **0.004106591129675507** |
| Minimum score | 0.0013807323994114995 |
| Mean score | 0.0023304136979970195 |
| Maximum score | 0.004892389755696058 |
| Normal calibration flags | **3 / 42 (7.14%)** |

Flagged filenames are **035.png, 055.png and 056.png**. The threshold and
calibration provenance are saved at full precision in `calibration/threshold.json`.
Every image's source path, full-precision score and decision are saved in
`calibration/validation_scores.csv`; all 42 recorded decisions match the rule.

These flags occur within a known-normal calibration set. This same validation
split was used for checkpoint selection, so the flagged fraction is not an
independent estimate for new normal images. It is not a reserved test result or
evidence of defect-detection performance.

#### Spatial error maps and common colour scale

For each of the three inspection images, the helper squares the differences
between the prepared input and latent-mean reconstruction, then averages across
RGB channels. This produces a finite **512 × 512 float32** spatial map.
Averaging its pixels should give the mean squared RGB image score.

| Image | Saved calibration score | Spatial map mean |
| --- | ---: | ---: |
| 001.png | 0.0023934857454150915 | 0.002393485978245735 |
| 087.png | 0.0022708468604832888 | 0.002270846627652645 |
| 206.png | 0.0015407655155286193 | 0.0015407652826979756 |

All **six checks** passed: each map mean matches both the directly computed RGB
score and the saved calibration score within `rtol=1e-5`, `atol=1e-8`.
The largest absolute difference is **2.3283064365386963e-10**. Small differences
come from floating-point calculations, including reductions and batch sizes.

All three figures use the same linear **magma** colour scale, with a numerical
colour bar and bounds **0.0 to 0.36739954352378845**. The upper bound is the
largest spatial squared error across these three E3 maps. There is no separate
normalization for each image and no probability interpretation of the scale.
Native 512 × 512 arrays are enlarged into figure panels using nearest-neighbour
interpolation; the saved PNGs are 3600 × 1440 pixels at 300 dpi. This display
enlargement does not change the arrays used for scoring or the numerical checks.

Visible differences are mainly around the rings, rim reflections and highlights
that the reconstructions smooth over. The figure captions state that colours
show **reconstruction differences, not defect probabilities or confirmed defect
locations**. All three source bottles are held-out normal images.

#### Output locations, provenance and completed Phase 4 work

The 12 new files under `outputs/E3/full_training/`, excluded from Git, are:

- `inspection/`: `loss_curves.png`, `loss_curves.pdf`, three labelled
  `<stem>_reconstruction.png` comparisons and `inspection_report.json`.
- `calibration/`: `validation_scores.csv` and `threshold.json`.
- `heatmaps/`: `001_heatmap.png`, `087_heatmap.png`, `206_heatmap.png` and
  `heatmap_report.json`.

The reports record actual sources, checkpoint/split hashes, dimensions, methods,
numerical checks, display settings and output paths. All helpers used the same
epoch-13 checkpoint and unchanged E1 manifest. The commands used were:

```powershell
.\.venv\Scripts\python.exe -B .\src\bottle_vae\inspect_training.py --run-folder .\outputs\E3\full_training
.\.venv\Scripts\python.exe -B .\src\bottle_vae\scoring.py --run-folder .\outputs\E3\full_training --percentile 95 --batch-size 8
.\.venv\Scripts\python.exe -B .\src\bottle_vae\heatmaps.py --run-folder .\outputs\E3\full_training
```

Calibration refuses to overwrite its existing results. To inspect saved figures
and reports without rerunning anything:

```powershell
Invoke-Item .\outputs\E3\full_training\inspection\loss_curves.png
Invoke-Item .\outputs\E3\full_training\inspection\001_reconstruction.png
Invoke-Item .\outputs\E3\full_training\heatmaps\001_heatmap.png
Get-Content .\outputs\E3\full_training\calibration\threshold.json
Import-Csv .\outputs\E3\full_training\calibration\validation_scores.csv
```

All three helpers exited successfully. Before/after hash checks confirmed all
**43 E1/E2 artifacts**, all **four E3 training files** (including its checkpoint)
and all **209 original normal training images** were unchanged. No reserved test
images or masks were accessed, no settings were tuned, and no temporary scripts,
synthetic data or new dependencies were used. Verification helpers remain.
Only this README and the 12 new derived E3 outputs were changed for this task.

**The agreed Phase 4 work is complete:** E1, E2 and E3 have their recorded training
results and computational cost, normal-image inspection, individual thresholds
and requested heatmaps. No ranking is made from raw total loss.
The subsequent Phase 5 image-level results are documented below; the paper and
demo remain Phase 6 and have not started.

## Phase 5: frozen image-level test evaluation

Completed on 2026-10-09 with the actual reserved test images. The model was
trained only on normal images; defect labels below are taken from the test
category folders solely for evaluation. No masks, retraining, threshold
recalculation, tuning or extra reconstruction previews were used.

### Actual test data and frozen provenance

Every source file was checked as PNG, verified and fully decoded before scoring:

| Test category | Expected | Actual files | Readable |
| --- | ---: | ---: | ---: |
| good | 20 | 20 | 20 |
| broken_large | 20 | 20 | 20 |
| broken_small | 22 | 22 | 22 |
| contamination | 21 | 21 | 21 |
| **Total** | **83** | **83** | **83** |

All were 900 × 900 RGB images. There were no count discrepancies, unexpected
entries or unreadable images. The evaluator reports count discrepancies and
uses actual counts; an unreadable file or unexpected entry stops evaluation
instead of silently skipping it. All configurations used the exact same ordered
test manifest, including relative paths, category labels and source hashes.
The byte-identical `dataset_report.json` copies have SHA-256:

```text
63ed8c260011c5954d6b5b8c84eee0eb462506ae1c521c7a366cb53e0e182480
```

| Configuration | Input / latent | Selected epoch | Loaded full-precision threshold |
| --- | --- | ---: | ---: |
| E1 | 128 × 128 / 16 | 47 | 0.0034759183879941686 |
| E2 | 256 × 256 / 32 | 45 | 0.0033652833313681184 |
| E3 | 512 × 512 / 64 | 13 | 0.004106591129675507 |

For every configuration, the loaded checkpoint's SHA-256, path, epoch and
dimensions match its recorded calibration provenance. The saved normal split
hash and training summary also match. Models load strictly on CPU with four
threads. The existing preprocessing converts originals to RGB, resizes them
bilinearly and produces float32 values in `[0, 1]`. Existing `score_images` runs
inference without gradients, decodes the latent mean, and averages squared
differences over all RGB pixels. KL is excluded from the anomaly score.

Each threshold is loaded directly from its existing `calibration/threshold.json`;
the evaluator does not call the percentile/calibration function. The decision
remains **score > threshold**; equality stays normal. The CSV labels are
**0 = normal, 1 = defective**, with defective as the positive class.

### Confusion matrices, detection metrics and recorded training cost

The confusion-matrix layout in each `metrics.json` is explicitly labelled:

```text
                         Predicted defective   Predicted normal
Actual defective                  TP                  FN
Actual normal                     FP                  TN

E1: [[26, 37], [1, 19]]
E2: [[33, 30], [2, 18]]
E3: [[25, 38], [1, 19]]
```

TP counts detected defects; FN counts missed defects. FP counts normal bottles
flagged defective, and TN counts normal bottles correctly left normal.
Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, F1 is
`2TP/(2TP+FP+FN)`, and the normal false-positive rate is `FP/(FP+TN)`.

This is the comparison saved at full precision in `outputs/phase5_comparison.csv`.
Display values below are rounded; the training time and actual epoch count come
from each existing full-training summary. Parameter counts are measured from
the strictly loaded models, including E1 whose older training summary did not
record a parameter count.

| Config | TP | FN | FP | TN | Precision | Recall | F1 | Normal FP rate | ROC-AUC | Actual epochs | Training seconds | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 26 | 37 | 1 | 19 | 0.9630 | 0.4127 | 0.5778 | 5% | 0.8556 | 50 | 508.301 | 747,395 |
| E2 | 33 | 30 | 2 | 18 | 0.9429 | 0.5238 | 0.6735 | 10% | 0.8627 | 50 | 864.442 | 3,524,515 |
| E3 | 25 | 38 | 1 | 19 | 0.9615 | 0.3968 | 0.5618 | 5% | 0.8230 | 23 | 1,605.680 | 25,642,979 |

Each matrix sums to **83**, with **63 actual defects** and **20 actual normal
images**. All metrics are defined on this dataset. For other class/decision
counts, a zero denominator produces an explicit JSON `null` and CSV blank,
with the undefined metric named in the report. F1 uses the count formula above,
so missing every actual defect gives F1 zero rather than an undefined value.

### Detected and missed defects by category

Each cell is **detected / missed**, using the same frozen decisions:

| Defect category | Actual defects | E1 | E2 | E3 |
| --- | ---: | --- | --- | --- |
| broken_large | 20 | 11 / 9 | 12 / 8 | 9 / 11 |
| broken_small | 22 | 5 / 17 | 9 / 13 | 6 / 16 |
| contamination | 21 | 10 / 11 | 12 / 9 | 10 / 11 |

Category counts reconcile with the dataset; category detected totals equal TP
and missed totals equal FN. The good category's FP/TN totals also match its
20 images. `metrics.json` includes the relative paths of missed defects;
`image_scores.csv` retains every decision, including normal false positives.
Small breaks have the lowest category recall in all three configurations.

### ROC method and numerical verification

ROC uses the **continuous anomaly scores**, not the binary predictions. Scores
are sorted descending and equal scores are moved together, so filename order
cannot affect tied-score handling. The curve starts at `(0, 0)` and ends at
`(1, 1)` when both classes exist. AUC is calculated by trapezoidal integration.
An independent comparison of all **63 × 20 = 1,260** defect/normal score pairs
checks that area, giving half credit when a defect and normal score tie.

The saved ROC-AUC values are:

- E1: **0.8555555555555555**.
- E2: **0.8626984126984126**.
- E3: **0.823015873015873**.

All three ROC endpoint and pairwise checks passed within `rtol=1e-12`,
`atol=1e-12`. There were no tied score groups in these actual results; the code
handles ties without adding mock images or scores. ROC-AUC would be explicitly
undefined if either actual class were absent.

The ROC figures mark the operating point of the original frozen threshold.
The ROC sweep describes score ranking; it is not used to select another
threshold. No test-informed threshold or model changes were made.

### Commands, outputs and preservation

The command run from the repository root was:

```powershell
.\.venv\Scripts\python.exe -B .\src\bottle_vae\evaluate.py
```

It evaluates E1, E2 and E3 together. It refuses any existing configuration's
evaluation directory or existing comparison CSV before reading/scoring images.
Repeating the command was checked: it raised `FileExistsError` as intended,
did not rescore images, and all **19 evaluation output files** remained unchanged.

Each `outputs/<experiment>/full_training/evaluation/` contains:

- `dataset_report.json`: actual counts, readability, ordered sources and hashes.
- `image_scores.csv`: relative source path, category, actual label, full-precision
  score and predicted label for all 83 images. Category paths distinguish images
  that share the same filename in different folders.
- `metrics.json`: labelled confusion matrix, metrics, category totals, missed
  source paths, undefined-metric policy and reconciliation checks.
- `provenance.json`: frozen checkpoint/calibration/split hashes, threshold,
  configuration, methods, CPU settings, training cost and scoring duration.
- `roc_curve.csv`: full-precision false-/true-positive rates and cumulative TP/FP.
- `roc_curve.png`: continuous-score ROC with the frozen operating point marked.

`outputs/phase5_comparison.csv` is the single comparison table, including
detection metrics, recorded training seconds, actual epochs and parameter counts.
All outputs are excluded from Git. Inspect them without rerunning evaluation:

```powershell
Import-Csv .\outputs\phase5_comparison.csv | Format-Table experiment,TP,FN,FP,TN,precision,recall,f1,roc_auc
Get-Content .\outputs\E1\full_training\evaluation\metrics.json
Import-Csv .\outputs\E2\full_training\evaluation\image_scores.csv
Invoke-Item .\outputs\E3\full_training\evaluation\roc_curve.png
```

The evaluation completed with process exit code 0. All **249 image scores**
(83 per configuration) are finite. There were no evaluation failures, count
discrepancies, missing decisions or failed provenance/reconciliation checks.
Before/after SHA-256 checks confirmed all **67 previous output files** and all
**292 normal-training/test source images** were unchanged. Checkpoints, recorded
thresholds and previous figures/reports are preserved. Masks were not accessed.
No synthetic data, temporary scripts, new dependencies or helper removals were
introduced.

Changed source files are `evaluate.py` (the working evaluation pipeline),
`scoring.py` (the existing checkpoint-loading code extracted into `load_model`)
and this README. Existing score computation and calibration mathematics remain
unchanged; no calibration command was rerun.

### What the results support, and their limitations

These results support a **qualified yes** to the research question: normal-only
VAE training followed by reconstruction error can detect some previously unseen
bottle defects, and the observed ROC-AUC values are above the chance reference
of 0.5. However, the frozen thresholds detect only **39.7–52.4%** of the defects
and miss **30–38 of 63**. High precision does not imply high defect coverage.
Normal-only training also does not guarantee zero normal flags: test false-positive
rates were 5%, 10% and 5%.

E2 has the highest observed recall, F1 and ROC-AUC among these three runs, while
flagging one more normal image than E1/E3. This is an observation on this test
set, not proof of a statistically significant or universal advantage. E3's
larger parameter count and recorded training cost did not yield better observed
detection metrics in this run. No ranking is made from raw total training loss.

Resolution, latent dimension and resulting parameter count change together,
so these are comparisons of **complete configurations**, not isolated tests of
one setting. Recorded runtime also reflects different actual epoch counts
(E3 stopped early), the same hardware and the existing timing convention.
Only one seed and one 83-image bottle test set were evaluated; uncertainty
intervals and significance tests were not computed. Precision depends on the
63-defect/20-normal class mix and is not a guaranteed deployment value.
Thresholds were calibrated on the normal validation split also used for
checkpoint selection. This is image-level evaluation; no claim about defect
localization or mask-level performance is made.

**Phase 5 image-level evaluation is complete. Phase 6 has not started.**

## Attribution and dataset licence

Dataset copyright 2019 MVTec Software GmbH.
Source: [official MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
MVTec AD is released under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Credit the creators, retain the licence, indicate modifications and follow the
noncommercial and share-alike conditions. Keep the bundled data/bottle/readme.txt
and data/bottle/license.txt. This dataset licence does not automatically license
the project's source code.

Citation from the distributed dataset:
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019.


