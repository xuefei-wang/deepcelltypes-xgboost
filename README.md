# XGBoost Baseline

Gradient boosted trees for cell type classification using mean intensity features (269 channels). Standard ML method -- no paper-specific implementation. Features are mean marker intensity per channel within each cell mask.

## Installation

Basic (baseline only):

```bash
pip install -e .
```

With Optuna hyperparameter tuning:

```bash
pip install -e ".[tuning]"
```

With all optional dependencies (Optuna + W&B):

```bash
pip install -e ".[all]"
```

## Usage

### Baseline

```bash
python -m xgb --model_name xgb_0 --split_file splits/fov_split_v7.json
```

### Hyperparameter Tuning

```bash
python -m xgb tune --study_name tune_0 --n_trials 100 --split_file splits/fov_split_v7.json
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `xgb_baseline_0` | Name for saving model/predictions |
| `--split_file` | (required) | Path to pre-computed FOV split JSON |
| `--zarr_dir` | `$DATA_DIR/tissuenet-caitlin-labels.zarr` | Path to TissueNet zarr archive |
| `--n_estimators` | 100 | Number of boosting rounds |
| `--max_depth` | 6 | Maximum tree depth |
| `--learning_rate` | 0.1 | Learning rate (eta) |
| `--min_channels` | 3 | Min non-DAPI channels per dataset |
| `--features_cache` | None | Path to cache extracted features (.npz) |
| `--enable_wandb` | False | Log to Weights & Biases |

For tuning, additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--study_name` | auto-generated | Name for the Optuna study |
| `--n_trials` | 100 | Number of tuning trials |
| `--metric` | `macro_accuracy` | Metric to optimize (`macro_accuracy` or `weighted_accuracy`) |
| `--max_tuning_samples` | 500000 | Subsample training data for faster tuning |
| `--storage` | None | Optuna storage URL for distributed tuning |
