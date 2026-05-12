"""
XGBoost hyperparameter tuning using Optuna.

Performs Bayesian optimization over core XGBoost hyperparameters:
- n_estimators, max_depth, learning_rate
- min_child_weight, subsample, colsample_bytree
"""

import os
import click
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb


# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))

from deepcell_types.training.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcell_types.training.baseline_features import extract_features_from_zarr, compute_baseline_metrics


class XGBoostObjective:
    """Optuna objective for XGBoost hyperparameter tuning."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        metric: str = "macro_accuracy",
        hierarchy: dict = None,
        ct2idx: dict = None,
        device: str = "cpu",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.metric = metric
        self.hierarchy = hierarchy
        self.ct2idx = ct2idx
        self.device = device

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to maximize
        """
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Fixed parameters
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "tree_method": "hist",
            "device": self.device,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Train model with early stopping
        early_stopping_rounds = max(10, params["n_estimators"] // 10)
        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_prob = model.predict_proba(self.X_val)

        metrics = compute_baseline_metrics(
            self.y_val, y_pred, y_prob, self.num_classes,
            hierarchy=self.hierarchy, ct2idx=self.ct2idx,
        )

        return metrics[self.metric]


def run_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    n_trials: int = 100,
    metric: str = "macro_accuracy",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    hierarchy: dict = None,
    ct2idx: dict = None,
    device: str = "cpu",
) -> Tuple[optuna.Study, dict]:
    """
    Run hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_classes: Number of classes
        n_trials: Number of trials to run
        metric: Metric to optimize
        study_name: Name for the Optuna study
        storage: Optional database URL for distributed tuning

    Returns:
        study: Optuna study object
        best_params: Best hyperparameters found
    """
    if study_name is None:
        study_name = f"xgboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Create objective
    objective = XGBoostObjective(
        X_train, y_train, X_val, y_val, num_classes, metric,
        hierarchy=hierarchy, ct2idx=ct2idx, device=device,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study, study.best_params


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    num_classes: int,
    device: str = "cpu",
    hierarchy: dict = None,
    ct2idx: dict = None,
) -> Tuple[xgb.XGBClassifier, dict]:
    """
    Train final model with best parameters.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        best_params: Best hyperparameters from tuning
        num_classes: Number of classes
        device: Device for training (cpu or cuda:N)
        hierarchy: Cell type hierarchy for hierarchical eval
        ct2idx: Cell type to index mapping

    Returns:
        model: Trained XGBoost model
        metrics: Test set metrics
    """
    params = {
        **best_params,
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
    }

    early_stopping_rounds = max(10, params.get("n_estimators", 100) // 10)
    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )
    print(f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.6f}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = compute_baseline_metrics(
        y_test, y_pred, y_prob, num_classes,
        hierarchy=hierarchy, ct2idx=ct2idx,
    )

    return model, metrics


@click.command()
@click.option("--study_name", type=str, default=None, help="Name for the Optuna study")
@click.option("--n_trials", type=int, default=100, help="Number of tuning trials")
@click.option(
    "--metric",
    type=click.Choice(["macro_accuracy", "weighted_accuracy"]),
    default="macro_accuracy",
    help="Metric to optimize",
)
@click.option("--enable_wandb", type=bool, default=False)
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR),
)
@click.option(
    "--skip_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to skip",
)
@click.option(
    "--keep_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to keep",
)
@click.option(
    "--storage",
    type=str,
    default=None,
    help="Optuna storage URL (e.g., sqlite:///tuning.db) for distributed tuning",
)
@click.option("--split_mode", type=click.Choice(["fov", "patch"]), default="fov",
              help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)")
@click.option("--split_file", type=str, default=None,
              help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)")
@click.option("--min_channels", type=int, default=3, help="Min non-DAPI channels per dataset (filters 2-channel datasets)")
@click.option("--max_tuning_samples", type=int, default=500000, help="Max training samples for tuning trials (subsample for speed)")
@click.option("--device_num", type=str, default="cpu", help="Device for XGBoost (cpu or cuda:N for GPU acceleration)")
def main(
    study_name: Optional[str],
    n_trials: int,
    metric: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    storage: Optional[str],
    split_mode: str,
    split_file: str,
    min_channels: int,
    max_tuning_samples: int,
    device_num: str,
):
    """Run XGBoost hyperparameter tuning with Optuna."""
    # Initialize wandb if enabled
    if enable_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project="deepcelltypes-temp-train",
            dir="wandb_tmp",
            job_type="tuning",
            name=study_name or f"xgb_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "xgboost_tuning",
                "n_trials": n_trials,
                "metric": metric,
                "split_mode": split_mode,
            },
        )

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(f"Optimizing: {metric}")
    print(f"Number of trials: {n_trials}")

    # Convert to lists
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError("--split_file is required. Generate one with: python -m scripts.generate_splits")

    # Extract features directly from zarr (fast path, ~20-50x faster than DataLoader)
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        min_channels=min_channels,
    )

    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_test, y_test = data["X_val"], data["y_val"]

    print(f"Training set: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Remap labels to contiguous 0-indexed (XGBoost requires this).
    # Train labels first so they get indices [0..n_train_classes-1],
    # then append any test-only labels after.
    train_unique = np.sort(np.unique(y_train_full))
    label_to_compact = {orig: i for i, orig in enumerate(train_unique)}
    next_idx = len(train_unique)
    for label in np.sort(np.unique(y_test)):
        if label not in label_to_compact:
            label_to_compact[label] = next_idx
            next_idx += 1
    n_classes_compact = next_idx
    compact_ct2idx = {
        name: label_to_compact[idx]
        for name, idx in dct_config.ct2idx.items()
        if idx in label_to_compact
    }

    y_train_full = np.array([label_to_compact[y] for y in y_train_full])
    y_test = np.array([label_to_compact[y] for y in y_test])
    print(f"Unique cell types in data: {n_classes_compact} (of {num_classes} total)")

    # Save full training data for final model (before subsampling)
    X_train_all = X_train_full
    y_train_all = y_train_full

    # Subsample training data for faster tuning trials
    if max_tuning_samples and len(X_train_full) > max_tuning_samples:
        subsample_idx = np.random.RandomState(42).choice(
            len(X_train_full), max_tuning_samples, replace=False
        )
        X_train_full = X_train_full[subsample_idx]
        y_train_full = y_train_full[subsample_idx]
        print(f"Subsampled training data to {max_tuning_samples} samples for tuning")

    # Split training into train/val (75/25 of training = 60/20 of total)
    # Use stratified split to ensure all classes appear in both partitions.
    # StratifiedShuffleSplit requires >=2 samples per class. Duplicate singletons.
    from sklearn.model_selection import StratifiedShuffleSplit
    unique, counts = np.unique(y_train_full, return_counts=True)
    singletons = unique[counts == 1]
    if len(singletons) > 0:
        singleton_mask = np.isin(y_train_full, singletons)
        singleton_idx = np.where(singleton_mask)[0]
        X_train_full = np.concatenate([X_train_full, X_train_full[singleton_idx]])
        y_train_full = np.concatenate([y_train_full, y_train_full[singleton_idx]])
        print(f"Duplicated {len(singletons)} singleton classes for stratified split")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_full, y_train_full))

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Run hyperparameter tuning
    print(f"\nStarting hyperparameter tuning ({n_trials} trials)...")
    study, best_params = run_tuning(
        X_train, y_train, X_val, y_val,
        num_classes=n_classes_compact,
        n_trials=n_trials,
        metric=metric,
        study_name=study_name,
        storage=storage,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
        device=device_num,
    )

    print(f"\nBest trial:")
    print(f"  Value ({metric}): {study.best_trial.value:.4f}")
    print(f"  Params: {best_params}")

    # Train final model on FULL training set (not subsampled) with best params
    print(f"\nTraining final model with best parameters on full data ({len(X_train_all)} samples)...")
    X_train_combined = X_train_all
    y_train_combined = y_train_all

    model, test_metrics = train_best_model(
        X_train_combined, y_train_combined,
        X_test, y_test,
        best_params, n_classes_compact,
        device=device_num,
        hierarchy=CELL_TYPE_HIERARCHY,
        ct2idx=compact_ct2idx,
    )

    print(f"\nFinal Test Results:")
    print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {test_metrics['weighted_accuracy']:.4f}")

    # Save results
    output_dir = Path("output/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name_safe = study_name or f"xgb_tuning_{timestamp}"

    # Save best parameters
    params_path = output_dir / f"{study_name_safe}_best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": study.best_trial.value,
            "metric": metric,
            "n_trials": n_trials,
            "test_metrics": {
                "macro_accuracy": test_metrics["macro_accuracy"],
                "weighted_accuracy": test_metrics["weighted_accuracy"],
            },
        }, f, indent=2)
    print(f"\nBest parameters saved to {params_path}")

    # Save model
    model_path = Path(f"models/xgb_tuned_{study_name_safe}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save study history
    history_path = output_dir / f"{study_name_safe}_history.csv"
    study.trials_dataframe().to_csv(history_path, index=False)
    print(f"Trial history saved to {history_path}")

    # Log to wandb if enabled
    if enable_wandb:
        wandb.log({
            "best_trial_value": study.best_trial.value,
            "test/macro_accuracy": test_metrics["macro_accuracy"],
            "test/weighted_accuracy": test_metrics["weighted_accuracy"],
            **{f"best_param/{k}": v for k, v in best_params.items()},
        })

        # Log optimization history plot
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            wandb.log({"optimization_history": fig})
        except Exception:
            pass

        # Log parameter importance
        try:
            fig = optuna.visualization.plot_param_importances(study)
            wandb.log({"param_importances": fig})
        except Exception:
            pass

        run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
