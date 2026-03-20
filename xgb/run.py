"""
XGBoost baseline for cell type classification.

Uses mean intensity per channel as features to classify cell types.
This provides a simple baseline to compare against the transformer-based model.
"""

import os
import click
import numpy as np
from pathlib import Path
from typing import Tuple
import xgboost as xgb

# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))

from deepcelltypes.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcelltypes.utils import (
    compute_baseline_metrics,
    save_baseline_predictions,
    extract_features_from_zarr,
)


@click.command()
@click.option("--model_name", type=str, default="xgb_baseline_0")
@click.option("--device_num", type=str, default="cuda:0", help="Unused, kept for consistency")
@click.option("--enable_wandb", type=bool, default=False)
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR / "tissuenet-caitlin-labels.zarr"),
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
    help="Dataset keys to keep (exclusive with skip_datasets)",
)
@click.option(
    "--n_estimators",
    type=int,
    default=100,
    help="Number of boosting rounds",
)
@click.option(
    "--max_depth",
    type=int,
    default=6,
    help="Maximum tree depth",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.1,
    help="Learning rate (eta)",
)
@click.option("--split_mode", type=click.Choice(["fov", "patch"]), default="fov",
              help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)")
@click.option("--split_file", type=str, default=None,
              help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)")
@click.option("--features_cache", type=str, default=None,
              help="Path to cache extracted features (.npz). Reuses cache if it exists.")
@click.option("--min_channels", type=int, default=3, help="Min non-DAPI channels per dataset (filters 2-channel datasets)")
def main(
    model_name: str,
    device_num: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    split_mode: str,
    split_file: str,
    features_cache: str,
    min_channels: int,
):
    """Train XGBoost baseline for cell type classification."""
    # Initialize wandb if enabled
    if enable_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project="deepcelltypes-temp-train",
            dir="wandb_tmp",
            job_type="train",
            name=f"{model_name}_xgboost",
            config={
                "model_type": "xgboost",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "split_mode": split_mode,
            },
        )

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError("--split_file is required. Generate one with: python -m scripts.generate_splits")

    # Extract features directly from zarr (fast path, no DataLoader overhead)
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        cache_path=features_cache,
        min_channels=min_channels,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_val"], data["y_val"]
    test_dataset_names = data["val_dataset_names"]
    test_fov_names = data["val_fov_names"]
    test_cell_indices = data["val_cell_indices"]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Remap labels to contiguous 0-indexed (XGBoost requires this).
    # Train labels must be contiguous [0..n_train-1]; test-only labels appended after.
    train_unique = np.sort(np.unique(y_train))
    label_to_compact = {orig: i for i, orig in enumerate(train_unique)}
    next_idx = len(train_unique)
    for label in np.sort(np.unique(y_test)):
        if label not in label_to_compact:
            label_to_compact[label] = next_idx
            next_idx += 1
    compact_to_label = {i: orig for orig, i in label_to_compact.items()}
    n_classes_compact = next_idx
    compact_ct2idx = {
        name: label_to_compact[idx]
        for name, idx in dct_config.ct2idx.items()
        if idx in label_to_compact
    }
    y_train_compact = np.array([label_to_compact[y] for y in y_train])
    y_test_compact = np.array([label_to_compact[y] for y in y_test])
    print(f"Unique cell types in data: {n_classes_compact} (of {num_classes} total)")

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    early_stopping_rounds = max(10, n_estimators // 10)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Filter eval_set to labels seen in training (test-only classes would be out of range)
    train_label_set = set(y_train_compact)
    eval_mask = np.isin(y_test_compact, list(train_label_set))
    if eval_mask.sum() < len(y_test_compact):
        print(f"  Note: {len(y_test_compact) - eval_mask.sum()} eval samples have train-unseen labels, excluded from eval_set")

    print(f"  Early stopping: {early_stopping_rounds} rounds")
    model.fit(
        X_train,
        y_train_compact,
        eval_set=[(X_test[eval_mask], y_test_compact[eval_mask])],
        verbose=True,
    )
    print(f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.6f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_compact = model.predict(X_test)
    y_prob_compact = model.predict_proba(X_test)  # (N, n_model_classes)

    # Metrics on compact labels (contiguous 0-indexed, required by confusion_matrix)
    metrics = compute_baseline_metrics(
        y_test_compact, y_pred_compact, y_prob_compact, n_classes_compact,
        hierarchy=CELL_TYPE_HIERARCHY, ct2idx=compact_ct2idx,
    )

    print(f"\nTest Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")

    # Log to wandb if enabled
    if enable_wandb:
        wandb.log({
            "test/macro_accuracy": metrics["macro_accuracy"],
            "test/weighted_accuracy": metrics["weighted_accuracy"],
        })

    # Save model
    model_path = Path(f"models/xgb_model_{model_name}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Map probabilities to ct2idx-sorted columns for saving
    # save_baseline_predictions expects y_prob with len(ct2idx) columns,
    # one per cell type sorted by ct2idx value
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob = np.zeros((len(y_test), len(dct_config.ct2idx)), dtype=np.float32)
    for compact_idx, orig_idx in compact_to_label.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]

    # Save predictions
    output_path = Path(f"output/{model_name}_xgb_prediction.csv")
    save_baseline_predictions(
        y_test,
        y_prob,
        test_cell_indices,
        test_dataset_names,
        test_fov_names,
        dct_config.ct2idx,
        output_path,
    )

    if enable_wandb:
        run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
