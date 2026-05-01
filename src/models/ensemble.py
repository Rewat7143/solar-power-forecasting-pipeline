"""
Stacked ensemble builder using non-negative linear regression.

Combines recursive backtest outputs from multiple individual models
into a meta-learner that produces weighted ensemble predictions.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation import regression_metrics


def build_stacked_ensemble_backtest(
    recursive_backtests: Dict[str, pd.DataFrame],
    candidate_models: Sequence[str],
    meta_train_fraction: float = 0.70,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, object]]:
    """
    Build a stacked ensemble from recursive backtest predictions.

    Args:
        recursive_backtests: Model name → backtest DataFrame mapping.
        candidate_models: Ordered list of model names to stack.
        meta_train_fraction: Fraction of forecast origins for meta-learner training.

    Returns:
        stacked: Ensemble backtest DataFrame.
        all_metrics: Metrics across all hours.
        daylight_metrics: Metrics for daylight hours only.
        stack_info: Ensemble metadata (members, weights, holdout metrics).
    """
    if len(candidate_models) < 2:
        raise ValueError("Need at least two candidate models for stacked ensemble.")

    merged = None
    feature_columns = []
    for idx, model_name in enumerate(candidate_models):
        pred_col = f"pred_{idx + 1}"
        feature_columns.append(pred_col)
        frame = recursive_backtests[model_name].loc[
            :,
            ["timestamp", "forecast_origin", "actual_power_kw", "is_daylight", "forecast_power_kw"],
        ].copy()
        frame = frame.rename(columns={"forecast_power_kw": pred_col})
        keep_cols = ["timestamp", "forecast_origin", pred_col]
        if merged is None:
            keep_cols = [
                "timestamp", "forecast_origin", "actual_power_kw", "is_daylight", pred_col,
            ]
        frame = frame.loc[:, keep_cols]
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=["timestamp", "forecast_origin"], how="inner")
        )

    if merged is None or merged.empty:
        raise ValueError("Stacked ensemble merge failed due to empty backtest frames.")

    merged = merged.sort_values(["forecast_origin", "timestamp"]).reset_index(drop=True)
    daylight = merged.loc[merged["is_daylight"] == 1].copy()
    if daylight.empty:
        raise ValueError("No daylight rows available for stacked ensemble fitting.")

    # Split forecast origins into meta-train and holdout
    unique_origins = np.sort(daylight["forecast_origin"].unique())
    split_idx = max(
        1, min(len(unique_origins) - 1, int(len(unique_origins) * meta_train_fraction)),
    )
    train_origins = set(unique_origins[:split_idx])
    holdout_mask = daylight["forecast_origin"].isin(unique_origins[split_idx:])
    if holdout_mask.sum() == 0:
        holdout_mask = np.ones(len(daylight), dtype=bool)

    X_train = daylight.loc[
        daylight["forecast_origin"].isin(train_origins), feature_columns
    ].to_numpy(dtype=float)
    y_train = daylight.loc[
        daylight["forecast_origin"].isin(train_origins), "actual_power_kw"
    ].to_numpy(dtype=float)

    if len(y_train) < 32:
        raise ValueError("Not enough daylight samples to fit stacked ensemble meta-learner.")

    meta_model = LinearRegression(fit_intercept=False, positive=True)
    meta_model.fit(X_train, y_train)

    # Predict on all rows
    all_pred = np.clip(
        meta_model.predict(merged[feature_columns].to_numpy(dtype=float)), 0.0, None,
    )
    stacked = merged.loc[
        :, ["timestamp", "forecast_origin", "actual_power_kw", "is_daylight"]
    ].copy()
    stacked["forecast_power_kw"] = all_pred

    all_metrics = regression_metrics(
        stacked["actual_power_kw"].to_numpy(),
        stacked["forecast_power_kw"].to_numpy(),
    )
    daylight_stacked = stacked.loc[stacked["is_daylight"] == 1].copy()
    daylight_metrics = regression_metrics(
        daylight_stacked["actual_power_kw"].to_numpy(),
        daylight_stacked["forecast_power_kw"].to_numpy(),
    )

    # Holdout evaluation
    holdout_df = daylight.loc[holdout_mask].copy()
    holdout_pred = np.clip(
        meta_model.predict(holdout_df[feature_columns].to_numpy(dtype=float)), 0.0, None,
    )
    holdout_metrics = regression_metrics(holdout_df["actual_power_kw"].to_numpy(), holdout_pred)

    raw_weights = np.asarray(meta_model.coef_, dtype=float)
    weights = {name: float(w) for name, w in zip(candidate_models, raw_weights)}
    stack_info = {
        "members": list(candidate_models),
        "weights": weights,
        "meta_train_fraction": float(meta_train_fraction),
        "holdout_daylight_metrics": holdout_metrics,
    }
    return stacked, all_metrics, daylight_metrics, stack_info
