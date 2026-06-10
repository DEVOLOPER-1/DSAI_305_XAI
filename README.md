# WhiteBox — Explainable TMB Prediction

> **check out our results and findings**: [full research paper](https://github.com/DEVOLOPER-1/XAI-in-efficacy-of-immunotherapy/blob/main/research_paper/paper.pdf)
> **Research pipeline for predicting Tumour Mutational Burden (TMB) from multimodal cancer data.**
> This is the single source of truth for the codebase. Read it once, top-to-bottom, the first time you set up.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Data Layout](#5-data-layout)
6. [Configuration System](#6-configuration-system)
7. [Available Models](#7-available-models)
8. [Running Experiments](#8-running-experiments)
9. [Weights & Biases Integration](#9-weights--biases-integration)
10. [Adding a New Model](#10-adding-a-new-model)
11. [Team Standards — Read Before You Code](#11-team-standards--read-before-you-code)
12. [Create Your Member Folder](#12-create-your-member-folder)
13. [Make Command Reference](#13-make-command-reference)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [Gotchas & FAQ](#15-gotchas--faq)

---

## 1. Project Overview

**Task:** Predict Tumour Mutational Burden (TMB) — a continuous regression target — from multimodal STAD (Stomach Adenocarcinoma) patient data.

**Modalities supported:**

| Modality | Description | Config flag |
|----------|-------------|-------------|
| Tabular | Clinical features, RNA-seq expression, miRNA | `modalities.tabular: true` |
| Image | Whole-Slide Images (WSI) — SVS or pre-extracted `.npy` patch features | `modalities.image: true` |
| Fusion | Combined tabular + image *(planned)* | `model.category: fusion` |

**Metrics reported per run:** RMSE · MAE · R² · Pearson r · C-Index · AUROC / AUPRC (when `risk_threshold` is set).

**Experiment tracking:** All runs are logged to a shared [Weights & Biases](https://wandb.ai) project — hyperparameters, metrics, and model artifacts.

---

## 2. Prerequisites

You need the following installed before anything else.

### 2.1 Python ≥ 3.13

```bash
python3 --version   # must print 3.13 or higher
```

Install from [python.org](https://www.python.org/downloads/) or on Ubuntu:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt install python3.13 python3.13-venv
```

### 2.2 uv — the dependency manager

`uv` replaces `pip + venv` with a single, dramatically faster tool:

| Old way | uv equivalent |
|---------|--------------|
| `python -m venv .venv` | `uv venv` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install package` | `uv pip install package` |

Install it once, globally:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

> The `uv.lock` file pins every transitive dependency so every teammate gets the exact same environment.

### 2.3 Git & make

```bash
git --version   # any recent version
make --version  # on Windows: install via Git for Windows or use WSL2
```

---

## 3. Quick Start

```bash
# 1. Clone
git clone https://github.com/DEVOLOPER-1/DSAI_305_XAI
cd DSAI_305_XAI

# 2. Create the virtual environment and install core dependencies
make setup

# 3. Activate the environment
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 4. (Optional) Install the deep-learning stack (PyTorch, torchvision, timm)
make setup-dl

# 5. (Optional) Install developer tools (pre-commit, jupyter, ipykernel)
uv pip install -e ".[dev]"

# 6. Log in to Weights & Biases (one-time per machine)
wandb login

# 7. Run your first experiment
python main.py --config configs/experiments/random_forest.yaml --mode train
```

After training completes, metrics are printed to stdout and logged to W&B.

---

## 4. Project Structure

```
DSAI_305_XAI/
│
├── pyproject.toml              ← Project metadata, dependencies, ruff config
├── uv.lock                     ← Pinned dependency graph (committed — do not edit manually)
├── Makefile                    ← All team commands (setup, lint, leaderboard…)
├── STANDARDS.md                ← Team conventions — READ THIS before coding
├── README.md                   ← You are here
├── main.py                     ← CLI entry point: train / eval / predict / info modes
│
├── configs/                    ← All experiment configs — never hardcode hyperparameters
│   ├── _base.yaml              ← Shared defaults (dataset paths, W&B entity, modalities)
│   └── experiments/
│       ├── random_forest.yaml      ← Random Forest baseline on clinical data
│       └── lasso_regressor.yaml    ← Lasso regression baseline on RNA-seq data
│
├── src/                        ← Shared source code — coordinate before modifying
│   ├── config.py               ← YAML loader + DotDict; merges _base.yaml + experiment
│   ├── data_loader.py          ← Multimodal patient dataset (tabular + WSI/features)
│   ├── train.py                ← Training loop — neural (AdamW) and sklearn (fit/predict)
│   ├── inference.py            ← Evaluation and prediction helpers
│   ├── utils.py                ← Metrics (RMSE, MAE, R², C-Index, AUROC), checkpoint IO
│   └── models/
│       ├── __init__.py         ← Model factory / registry (get_model, list_models)
│       ├── tabular.py          ← DecisionTree, RandomForest, GradientBoostedTrees, Lasso
│       ├── image.py            ← ResNetEncoder stub (placeholder; full impl pending)
│       └── baselines.py        ← Legacy reference implementations
│
├── data/                       ← Patient data (large files gitignored — see §5)
│   ├── raw/                    ← Original downloaded files
│   └── processed/              ← Preprocessed parquet/CSV files ready for training
│
├── members/                    ← One folder per team member (generate with `make new-member`)
│   └── {your_name}/
│       ├── README.md           ← Your personal experiment log
│       ├── best_experiment.ipynb
│       └── experiments/
│           └── YYYY-MM-DD_description.ipynb
│
├── logs/
│   ├── leaderboard.csv         ← Team results (committed — update after every run)
│   └── runs/                   ← Per-run checkpoints and logs (gitignored)
│
└── cookiecutter-member/        ← Template for `make new-member` — do not edit manually
```

**Key split:**

| Directory | Who owns it | Rule |
|-----------|-------------|------|
| `src/` | Whole team | Discuss changes before pushing; PRs required for registry edits |
| `configs/experiments/` | You | One new YAML per distinct experiment |
| `members/{your_name}/` | You | Free sandbox — commit and experiment freely |

---

## 5. Data Layout

Large data files are **not committed** (gitignored). Prepare your `data/` folder as follows:

```
data/
├── raw/
│   ├── stad_clinical_patient.parquet   ← raw STAD clinical features + TMB target
│   └── stad_tmb.parquet                ← standalone TMB values by PATIENT_ID
│
└── processed/
    ├── stad_clinical_patient_selected.parquet   ← feature-selected clinical data (21 features)
    ├── stad_rna.parquet                         ← RNA-seq expression matrix (produced locally)
    └── stad_clinical_patient.parquet            ← full processed clinical parquet
```

> **Column requirements** for any tabular file:
> - Must contain a `PATIENT_ID` column (configurable via `dataset.patient_id_col`)
> - Must contain the regression target column, default `TMB` (configurable via `dataset.target_col`)

For **image modality**, place files in:
```
data/
├── images/<PATIENT_ID>/slide.svs       ← whole-slide SVS/TIFF files
└── features/<PATIENT_ID>.npy           ← pre-extracted patch features (N_patches, feature_dim)
```
Set `dataset.use_preextracted: true` (default) to load `.npy` files — no OpenSlide required.
Set `dataset.use_preextracted: false` to tile SVS files on-the-fly (requires `uv pip install openslide-python`).

---

## 6. Configuration System

All experiments are driven by YAML configs. No hardcoded hyperparameters in `.py` files.

### How configs are merged

```
configs/_base.yaml          ← loaded first (shared team defaults)
configs/experiments/X.yaml  ← deep-merged on top (your keys win)
```

`src/config.py` performs the merge and returns a `DotDict` so you can write `cfg.model.type` instead of `cfg["model"]["type"]`.

### Key config sections

```yaml
# ── Modalities ──────────────────────────────────────────────────────────────
modalities:
  tabular: true    # load clinical/genomic CSV or parquet
  image:   false   # load WSI tiles or pre-extracted .npy features

# ── Dataset ─────────────────────────────────────────────────────────────────
dataset:
  data_root:        data/processed
  tabular_file:     stad_clinical_patient_selected.parquet
  target_col:       TMB              # regression target column name
  patient_id_col:   PATIENT_ID
  val_ratio:        0.15             # fraction of patients held out for validation
  seed:             42
  batch_size:       32
  use_preextracted: true             # true = load .npy; false = tile SVS on-the-fly

# ── Model ────────────────────────────────────────────────────────────────────
model:
  category:   tabular_only           # tabular_only | image_only | fusion
  type:       random_forest          # see §7 for all available types
  # model-specific hyperparameters follow…

# ── Training ────────────────────────────────────────────────────────────────
training:
  save_dir:       logs/runs/checkpoints
  risk_threshold: 3.84   # TMB threshold for AUROC/AUPRC; set to null to skip

# ── W&B ─────────────────────────────────────────────────────────────────────
wandb:
  entity:    mohamed-mourad-zewail-city
  project:   TMB-prediction
  run_name:  my_experiment_v1
  tags:
    - tabular_only
    - random_forest
```

### Creating a new experiment config

```bash
cp configs/experiments/random_forest.yaml configs/experiments/ahmed_rf_v2.yaml
# Edit ahmed_rf_v2.yaml — change model.type, hyperparams, dataset.tabular_file, etc.
python main.py --config configs/experiments/ahmed_rf_v2.yaml --mode train
```

Config naming convention: `{method}_{variant}.yaml` — lowercase, underscores.

---

## 7. Available Models

### Tabular models (`model.category: tabular_only`)

| `model.type` | Class | Key hyperparameters |
|---|---|---|
| `lasso_regressor` | `LassoRegressor` | `alpha`, `max_iter`, `selection`, `tol` |
| `decision_tree` | `DecisionTreeRegressor` | `max_depth`, `min_samples_split`, `criterion` |
| `random_forest` | `RandomForestRegressor` | `n_estimators`, `max_depth`, `max_features`, `oob_score` |
| `gradient_boosted` | `GradientBoostedTrees` | `max_iter`, `learning_rate`, `max_depth`, `early_stopping` |

All tabular models share the same sklearn-compatible API (`fit` / `predict`) and support `print_feature_importances()` to inspect top-N features.

### Image models (`model.category: image_only`)

| `model.type` | Status | Notes |
|---|---|---|
| `resnet` | Stub (no-op) | Returns image input unchanged; full implementation pending |

### Fusion models (`model.category: fusion`)

Fusion models are **not yet implemented**. Registry entries are reserved for future work.

---

## 8. Running Experiments

The CLI entry point is `main.py`. All modes require `--config`.

### Train

```bash
python main.py --config configs/experiments/random_forest.yaml --mode train
```

Trains the model, validates every `val_every` epochs (or once for tree models), saves the best checkpoint, and logs everything to W&B.

### Evaluate on the validation split

```bash
python main.py --config configs/experiments/random_forest.yaml --mode eval
```

Prints RMSE, MAE, R², Pearson r, C-Index (and AUROC/AUPRC if `risk_threshold` is set).

### Generate a prediction CSV

```bash
python main.py --config configs/experiments/random_forest.yaml --mode predict \
               --output submission.csv
```

### Inspect the resolved config and registered models

```bash
python main.py --config configs/experiments/random_forest.yaml --mode info
```

Prints the fully-merged config as JSON and lists all registered model types — useful for debugging config inheritance.

### Console-script aliases (after `uv pip install -e .`)

```bash
train-tracker --config configs/experiments/random_forest.yaml
eval-tracker  --config configs/experiments/random_forest.yaml
```

### Logging level

Add `--log-level DEBUG` to any command for verbose output:

```bash
python main.py --config configs/experiments/random_forest.yaml --mode train --log-level DEBUG
```

---

## 9. Weights & Biases Integration

Every run is automatically logged to the shared W&B project `TMB-prediction` under the entity `mohamed-mourad-zewail-city`.

### One-time setup (per machine)

1. Create an account at [wandb.ai](https://wandb.ai/site) using your university email.
2. Generate an API key from the **Settings** page.
3. Run (with your `.venv` activated):
   ```bash
   wandb login
   ```
4. Paste your API key when prompted.

> **Never** commit your API key to Git. It lives in `~/.netrc` after `wandb login`.

### What gets logged

| What | Where in W&B | When |
|------|-------------|------|
| Full merged config | `run.config` | Once at run init |
| Train loss + val metrics | `wandb.log(...)` | Every `val_every` epochs |
| Best val metrics | `run.summary` | End of run |
| Model artifact (`.pkl`) | Artifact | If `training.upload_pickled_model: true` |

---

## 10. Adding a New Model

1. **Create `src/models/my_model.py`** — export one class that:
   - Inherits from `_TabularBase` (tabular) or implements `__init__(cfg)` + `__call__(image, tabular)`.
   - Reads all hyperparameters from `cfg.model.*`.

2. **Add a loader function** in `src/models/__init__.py`:
   ```python
   def _load_my_model(cfg: DotDict) -> Any:
       from src.models.my_model import MyModel
       return MyModel(cfg)
   ```

3. **Register it** in the appropriate sub-registry:
   ```python
   _TABULAR_REGISTRY["my_model"] = _load_my_model
   ```

4. **Add a config** in `configs/experiments/my_model.yaml` with `model.type: my_model`.

5. **Open a PR** — per `STANDARDS.md §5`, registry changes require team review.

---

## 11. Team Standards — Read Before You Code

> 📋 **[`STANDARDS.md`](STANDARDS.md) is the team contract.** Read it before writing a single line of code.

Key rules:

| Topic | Rule |
|-------|------|
| **Branches** | `{name}/feature` or `{name}/fix` — never commit to `main` directly |
| **Commit messages** | `feat:`, `exp:`, `fix:`, `refactor:`, `docs:` prefixes |
| **Notebooks** | `YYYY-MM-DD_kebab-case-description.ipynb` — always date-prefixed |
| **`src/` changes** | Discuss in team chat first; PRs required for `__init__.py` |
| **Leaderboard** | Keep only your best 2 rows; keep file sorted |

---

## 12. Create Your Member Folder

```bash
make new-member
```

You will be prompted:

```
member_name  [your_name]:      ahmed
full_name    [Your Full Name]: Ahmed Mohsen
focus_area   [e.g. tabular, image, fusion]: tabular
```

This creates `members/ahmed/` with the standard structure. Then:

1. Fill in `members/ahmed/README.md` with your name and focus area.
2. Commit and push your empty folder:

```bash
git checkout -b ahmed/setup
git add members/ahmed/
git commit -m "feat: add ahmed member folder"
git push -u origin ahmed/setup
```

---

## 13. Make Command Reference

| Command | What it does |
|---------|-------------|
| `make setup` | Create `.venv` and sync all core dependencies from `uv.lock` |
| `make setup-dl` | Install PyTorch, torchvision, timm, einops |
| `make lint` | Run `ruff check src/` — style and bug warnings |
| `make format` | Run `ruff format src/` + `ruff check --fix src/` — auto-fix |
| `make new-member` | Generate `members/{name}/` via cookiecutter |
| `make lb` | Print the team leaderboard sorted by final score |
| `make submit FILE=… MSG=…` | Submit a prediction CSV |

---

## 14. Evaluation Metrics

| Metric | Key in results dict | Description |
|--------|---------------------|-------------|
| RMSE | `rmse` | Primary metric — lower is better |
| MAE | `mae` | Mean absolute error in TMB units |
| R² | `r2` | Coefficient of determination (1.0 = perfect) |
| Pearson r | `pearson_r` | Linear correlation with ground truth |
| C-Index | `c_index` | Survival concordance index (0.5 = random) |
| AUROC | `auroc` | Only computed when `training.risk_threshold` is set |
| AUPRC | `auprc` | Only computed when `training.risk_threshold` is set |

---

## 15. Gotchas & FAQ

**Q: `make setup` fails with "uv: command not found"**
→ Install `uv` first (see §2.2), then open a fresh terminal.

**Q: `FileNotFoundError` for my tabular file at startup**
→ The parquet/CSV files are not committed. Check `dataset.tabular_file` in your config and ensure the file exists under `dataset.data_root`.

**Q: `wandb` not installed warning during training**
→ Metrics will still print to stdout. Install W&B with `uv pip install wandb` and run `wandb login`.

**Q: `ruff` is not found after `make setup`**
→ `ruff` is in the `dev` optional group. Run `uv pip install -e ".[dev]"` or activate your venv first.

**Q: Tree model crashes with `model.parameters()` AttributeError**
→ Ensure `model.category: tabular_only` is set in your config. The training loop uses `model.type` to decide between `fit()` (sklearn) and gradient descent (PyTorch).

**Q: My notebook's output is huge and slowing down git**
→ Strip outputs before committing: `jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb`

**Q: Should I use `uv pip install` or just `pip install`?**
→ Always `uv pip install` inside this project to respect the `uv.lock` pins.

**Q: I changed `pyproject.toml` — do I need to re-run setup?**
→ Run `uv sync` (update environment) and `uv lock` (regenerate lock file), then commit both.

**Q: The deep-learning stack fails to install**
→ PyTorch can need a platform-specific URL for CUDA. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/), install torch manually, then run `make setup-dl`.

**Q: Can I add a new Python package for my experiment?**
→ Add it to `pyproject.toml` (`dependencies` if everyone needs it, `[dl]` if it requires torch). Run `uv lock` and commit both `pyproject.toml` and `uv.lock`.

---

*Team WhiteBox · Data Science & Artificial Intelligence · Zewail City of Science and Technology*
