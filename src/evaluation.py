"""
Evaluation metrics, calibration, model export, and persistence.

Provides MAE/RMSE/R² computation, affine calibration for bias
correction, model directory naming, and full model + metrics export.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for regression predictions."""
    def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
        try:
            return float(mean_squared_error(a, b, squared=False))
        except TypeError:
            return float(np.sqrt(mean_squared_error(a, b)))

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": compute_rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_affine_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Fit a simple y = slope * pred + intercept calibration on validation data."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid_mask.sum() < 2:
        return {"slope": 1.0, "intercept": 0.0}

    design = np.column_stack([y_pred[valid_mask], np.ones(valid_mask.sum())])
    slope, intercept = np.linalg.lstsq(design, y_true[valid_mask], rcond=None)[0]
    slope = float(np.clip(slope, 0.0, 3.0))
    return {"slope": slope, "intercept": float(intercept)}


def apply_affine_calibrator(
    preds: np.ndarray,
    calibrator: Optional[Dict[str, float]],
) -> np.ndarray:
    """Apply an affine calibrator to raw predictions, clipping negatives to zero."""
    arr = np.asarray(preds, dtype=float)
    if not calibrator:
        return np.clip(arr, 0.0, None)
    slope = float(calibrator.get("slope", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return np.clip(slope * arr + intercept, 0.0, None)


def sanitize_model_dir_name(model_name: str) -> str:
    """Convert a model display name to a safe filesystem directory name."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "model"


def export_models_and_metrics(
    model_entries: Dict[str, Dict[str, object]],
    ensemble_definition: Dict[str, object],
    config: Dict[str, object],
    one_step_metrics_df: pd.DataFrame,
    recursive_metrics_df: pd.DataFrame,
    latest_metrics_path: Path,
    models_dir: Path,
) -> None:
    """Persist trained models as .pkl and write manifest + metrics JSON."""
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_models = {}
    for model_name, entry in model_entries.items():
        folder_name = sanitize_model_dir_name(model_name)
        model_folder = models_dir / folder_name
        model_folder.mkdir(parents=True, exist_ok=True)
        model_obj = entry.get("model")
        if model_obj is not None:
            joblib.dump(model_obj, model_folder / "model.pkl")

        saved_models[model_name] = {
            "name": model_name,
            "family": str(entry.get("family", "unknown")),
            "model_type": type(model_obj).__name__ if model_obj is not None else "unknown",
            "has_calibrator": bool(entry.get("calibrator")),
            "saved_path": f"models/{folder_name}",
        }

    manifest = {
        "saved_models": saved_models,
        "ensemble_definition": ensemble_definition or {},
        "config": {
            "sequence_length": int(config["sequence_length"]),
            "feature_columns": list(config.get("feature_columns", [])),
        },
    }
    (models_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    # Build metrics payload
    one_step_records = one_step_metrics_df.sort_values("RMSE").to_dict(orient="records")
    daylight_records = (
        recursive_metrics_df.loc[recursive_metrics_df["Scope"] == "Daylight only"]
        .sort_values("RMSE")
        .drop(columns=["Scope"])
        .to_dict(orient="records")
    )

    latest_payload = {
        "run_date": pd.Timestamp.now().date().isoformat(),
        "source": "solar_power_forecasting_pipeline",
        "one_step": one_step_records,
        "recursive_daylight": daylight_records,
    }
    if ensemble_definition:
        holdout = ensemble_definition.get("holdout_daylight_metrics")
        weights = ensemble_definition.get("weights")
        if isinstance(holdout, dict):
            latest_payload["stacked_holdout_daylight"] = holdout
        if isinstance(weights, dict):
            latest_payload["stacked_weights"] = weights

    latest_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    latest_metrics_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")
