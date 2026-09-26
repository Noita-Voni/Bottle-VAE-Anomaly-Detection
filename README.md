# Unsupervised Industrial Bottle Defect Detection Using a Convolutional VAE

Advanced AI research question: can a convolutional variational autoencoder trained
only on normal MVTec AD bottle images detect defects using reconstruction error?

Phase 1 sets up the repository, Python environment and dataset checks.
No VAE, training, preprocessing or anomaly threshold is implemented.
Future training must use only normal training images. Test images and ground-truth
masks are reserved for evaluation; do not tune the model on the test set.

## Files

```text
bottle-vae-anomaly-detection/
├── README.md
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
│   └── data_check.py
├── notebooks/
│   └── .gitkeep
├── tests/
│   └── test_data_check.py
└── outputs/                    # ignored
```

- README.md explains scope, setup, data and commands.
- requirements.txt records that Phase 1 has no third-party runtime dependencies.
- pyproject.toml defines package metadata, the src layout, Python >=3.10,
  and an optional installed command. Setuptools is only a packaging dependency.
- .gitignore excludes the dataset, images, archives, virtual environments,
  caches, credentials and generated results.
- src/bottle_vae/__init__.py marks the Python package.
- src/bottle_vae/data_check.py checks required folders, PNG file counts,
  PNG signatures and one-to-one defect image/mask filename matches.
- tests/test_data_check.py uses standard-library unittest with temporary fixtures.
- notebooks/.gitkeep keeps an empty notebook directory in Git.
- data/ holds the original local data. outputs/ is reserved for future results.
  These directories are not distributed through Git and must be created after cloning.

## Windows PowerShell setup

Run from the repository root, using Python 3.10 or newer:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe --version
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

## Run the check

No package installation or network access is needed:

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

Run the synthetic checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Dataset and proposal counts

Normal training: proposal 209. Normal test: proposal 20.
Defective test total: proposal 63. These are comparison targets, not assumed facts.
The executable checker reports observed counts on every run.
Defect masks are counted separately from test images.
A defect image such as test/broken_large/000.png must have
ground_truth/broken_large/000_mask.png. Normal images do not need masks.

Checked on 2026-09-26 with Python 3.13.2: the real dataset passed all Phase 1
checks, and all six synthetic unit tests passed.

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

## GitHub

The local repository uses main. Before publishing, inspect the staged paths:

```powershell
git status --short
git diff --cached --stat
git ls-files data outputs .venv
```

The last command must print nothing. Create an empty GitHub repository under your
chosen account, then connect and push (replace YOUR_ACCOUNT):

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/bottle-vae-anomaly-detection.git
git push -u origin main
```

GitHub publication requires an account, visibility choice and authentication.
