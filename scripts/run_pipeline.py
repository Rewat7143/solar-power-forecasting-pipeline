#!/usr/bin/env python3
"""
Solar Power Forecasting Pipeline — Main Entry Point.

Orchestrates the full end-to-end workflow:
  1. Load solar meter data (local Excel or Google Sheets)
  2. Fetch and merge NASA POWER weather features
  3. Engineer features and split train / val / test
  4. Train tree-based models (Random Forest, optional LightGBM)
  5. Train deep-learning sequence models (CNN+LSTM, PatchTST, TFT, CNN+Transformer)
  6. Run rolling recursive backtests
  7. Build a stacked ensemble from backtest predictions
  8. Export models, metrics, and produce the final 24-hour forecast

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONFIG,
    FORECAST_STEPS,
    LIGHTGBM_ENABLED,
    STEPS_PER_HOUR,
    WEATHER_COLUMNS,
)
from src.data_loader import load_solar_dataset
from src.evaluation import (
    apply_affine_calibrator,
    export_models_and_metrics,
    fit_affine_calibrator,
    regression_metrics,
)
from src.features import (
    chronological_split,
    engineer_features,
    split_full_frame_by_cutoffs,
)
from src.forecasting import (
    get_future_weather,
    recursive_forecast,
    run_recursive_backtest,
)
from src.models.ensemble import build_stacked_ensemble_backtest
from src.models.sequence_models import build_sequence_model
from src.training import (
    build_scaled_sequences,
    predict_sequence_model,
    time_series_cv_search,
    train_sequence_model,
)
from src.visualization import (
    plot_future_forecast,
    plot_model_comparison,
    plot_recursive_backtest,
)
from src.weather import (
    align_weather_to_index,
    fetch_nasa_power_hourly,
    merge_solar_and_weather,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)


def main() -> None:
    """Run the full solar forecasting pipeline."""

    # ── 1. Data Ingestion ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading solar meter data")
    print("=" * 60)

    solar_df = load_solar_dataset(CONFIG)
    print(f"  Solar rows: {len(solar_df)}")
    print(f"  Time span:  {solar_df['timestamp'].min()} → {solar_df['timestamp'].max()}")
    print(f"  Max power:  {solar_df['power_kw'].max():.3f} kW")

    # ── 2. Weather Data ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Fetching NASA POWER weather data")
    print("=" * 60)

    weather_hourly = fetch_nasa_power_hourly(
        latitude=CONFIG["latitude"],
        longitude=CONFIG["longitude"],
        start_date=CONFIG["date_range"][0],
        end_date=CONFIG["date_range"][1],
        timezone=CONFIG["timezone"],
        time_standard=CONFIG["nasa_time_standard"],
        parameters=WEATHER_COLUMNS,
    )
    weather_5min = align_weather_to_index(
        weather_hourly, pd.DatetimeIndex(solar_df["timestamp"]),
    )
    merged_df = merge_solar_and_weather(solar_df, weather_5min)
    print(f"  Merged rows: {len(merged_df)}")

    # ── 3. Feature Engineering ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Engineering features")
    print("=" * 60)

    feature_df_full, model_df, feature_columns = engineer_features(
        merged_df=merged_df,
        night_power_threshold_kw=CONFIG["night_power_threshold_kw"],
        day_irradiance_threshold=CONFIG["day_irradiance_threshold"],
        daylight_hour_window=CONFIG["daylight_hour_window"],
    )
    train_df, val_df, test_df = chronological_split(
        model_df,
        train_fraction=CONFIG["train_fraction"],
        val_fraction=CONFIG["val_fraction"],
    )
    full_train_df, full_val_df, full_test_df = split_full_frame_by_cutoffs(
        full_frame=feature_df_full,
        train_end_ts=train_df["timestamp"].max(),
        val_end_ts=val_df["timestamp"].max(),
        required_columns=feature_columns,
    )
    print(f"  Daylight modelling rows: {len(model_df)}")
    print(f"  Train / Val / Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"  Night anomalies zeroed: {int(feature_df_full['night_anomaly_flag'].sum())}")
    print(f"  Features: {len(feature_columns)}")

    # ── 4. Tree-Based Models ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training tree-based models")
    print("=" * 60)

    model_entries = {}
    one_step_predictions = {}
    one_step_rows = []

    tree_model_configs = {"Random Forest": CONFIG["rf_param_grid"]}
    if LIGHTGBM_ENABLED:
        tree_model_configs["LightGBM"] = CONFIG["lgbm_param_grid"]

    for model_name, param_grid in tree_model_configs.items():
        model, cv_results = time_series_cv_search(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            target_col="power_kw",
            param_grid=param_grid,
            random_seed=CONFIG["random_seed"],
        )
        val_preds_raw = model.predict(val_df[feature_columns])
        calibrator = fit_affine_calibrator(val_df["power_kw"].to_numpy(), val_preds_raw)
        preds = apply_affine_calibrator(model.predict(test_df[feature_columns]), calibrator)
        metrics = regression_metrics(test_df["power_kw"].to_numpy(), preds)
        model_entries[model_name] = {
            "name": model_name, "family": "tree",
            "model": model, "calibrator": calibrator,
        }
        one_step_predictions[model_name] = preds
        one_step_rows.append({"Model": model_name, **metrics})
        print(f"  {model_name}: {metrics}")

    # ── 5. Sequence Models ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Training sequence models")
    print("=" * 60)

    sequence_bundle = build_scaled_sequences(
        frame=model_df,
        feature_columns=feature_columns,
        target_col="power_kw",
        train_size=len(train_df),
        val_size=len(val_df),
        sequence_length=CONFIG["sequence_length"],
    )

    seq_model_names = ["CNN + LSTM", "PatchTST", "Temporal Fusion Transformer", "CNN + Transformer"]
    for model_name in seq_model_names:
        seq_model = build_sequence_model(
            model_name=model_name,
            n_features=len(feature_columns),
            seq_len=CONFIG["sequence_length"],
            config=CONFIG,
        )
        seq_model, history_df = train_sequence_model(
            model=seq_model,
            bundle=sequence_bundle,
            batch_size=CONFIG["dl_batch_size"],
            learning_rate=CONFIG["dl_learning_rate"],
            weight_decay=CONFIG["dl_weight_decay"],
            max_epochs=CONFIG["dl_max_epochs"],
            patience=CONFIG["dl_patience"],
        )
        val_preds_scaled = predict_sequence_model(
            seq_model, sequence_bundle.X_sequences[sequence_bundle.val_mask],
        )
        val_preds_raw = sequence_bundle.target_scaler.inverse_transform(
            val_preds_scaled.reshape(-1, 1),
        ).reshape(-1)
        calibrator = fit_affine_calibrator(val_df["power_kw"].to_numpy(), val_preds_raw)

        preds_scaled = predict_sequence_model(
            seq_model, sequence_bundle.X_sequences[sequence_bundle.test_mask],
        )
        preds_raw = sequence_bundle.target_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1),
        ).reshape(-1)
        preds = apply_affine_calibrator(preds_raw, calibrator)
        metrics = regression_metrics(test_df["power_kw"].to_numpy(), preds)

        model_entries[model_name] = {
            "name": model_name, "family": "sequence",
            "model": seq_model,
            "feature_scaler": sequence_bundle.feature_scaler,
            "target_scaler": sequence_bundle.target_scaler,
            "calibrator": calibrator,
        }
        one_step_predictions[model_name] = preds
        one_step_rows.append({"Model": model_name, **metrics})
        print(f"  {model_name}: {metrics}")

    one_step_metrics_df = pd.DataFrame(one_step_rows).sort_values("RMSE").reset_index(drop=True)

    # ── 6. Recursive Backtesting ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Running recursive backtests")
    print("=" * 60)

    backtest_horizon_steps = CONFIG["backtest_horizon_hours"] * STEPS_PER_HOUR
    backtest_stride_steps = CONFIG["backtest_stride_hours"] * STEPS_PER_HOUR

    recursive_backtests = {}
    recursive_rows = []
    ensemble_definition = {}

    for model_name in model_entries.keys():
        model_entry = model_entries[model_name]
        backtest_df, all_metrics, daylight_metrics = run_recursive_backtest(
            model_entry=model_entry,
            full_history_df=feature_df_full,
            evaluation_df=full_test_df,
            feature_columns=feature_columns,
            sequence_length=CONFIG["sequence_length"],
            horizon_steps=backtest_horizon_steps,
            stride_steps=backtest_stride_steps,
        )
        backtest_df["forecast_power_kw"] = apply_affine_calibrator(
            backtest_df["forecast_power_kw"].to_numpy(), model_entry.get("calibrator"),
        )
        all_metrics = regression_metrics(
            backtest_df["actual_power_kw"].to_numpy(),
            backtest_df["forecast_power_kw"].to_numpy(),
        )
        daylight_df = backtest_df.loc[backtest_df["is_daylight"] == 1].copy()
        daylight_metrics = regression_metrics(
            daylight_df["actual_power_kw"].to_numpy(),
            daylight_df["forecast_power_kw"].to_numpy(),
        )
        recursive_backtests[model_name] = backtest_df
        recursive_rows.extend([
            {"Model": model_name, "Scope": "All hours", **all_metrics},
            {"Model": model_name, "Scope": "Daylight only", **daylight_metrics},
        ])
        print(f"  {model_name} daylight: {daylight_metrics}")

    # ── 7. Stacked Ensemble ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Building stacked ensemble")
    print("=" * 60)

    base_daylight = pd.DataFrame(recursive_rows)
    base_daylight = base_daylight.loc[base_daylight["Scope"] == "Daylight only"].sort_values("RMSE")
    top_k_models = base_daylight["Model"].head(int(CONFIG.get("stacking_top_k", 4))).tolist()

    if len(top_k_models) >= 2:
        ensemble_name = f"Stacked Ensemble ({' + '.join(top_k_models)})"
        try:
            ensemble_backtest, ens_all, ens_daylight, stack_info = build_stacked_ensemble_backtest(
                recursive_backtests=recursive_backtests,
                candidate_models=top_k_models,
                meta_train_fraction=float(CONFIG.get("stacking_meta_train_fraction", 0.70)),
            )
            recursive_backtests[ensemble_name] = ensemble_backtest
            recursive_rows.extend([
                {"Model": ensemble_name, "Scope": "All hours", **ens_all},
                {"Model": ensemble_name, "Scope": "Daylight only", **ens_daylight},
            ])
            ensemble_definition = {
                "name": ensemble_name,
                "method": "non_negative_linear_stacking",
                "members": stack_info["members"],
                "weights": stack_info["weights"],
                "meta_train_fraction": stack_info["meta_train_fraction"],
                "holdout_daylight_metrics": stack_info["holdout_daylight_metrics"],
            }
            print(f"  Ensemble daylight: {ens_daylight}")
            print(f"  Weights: {ensemble_definition['weights']}")
        except Exception as exc:
            print(f"  Stacked ensemble skipped: {exc}")

    recursive_metrics_df = pd.DataFrame(recursive_rows).sort_values(
        ["Scope", "RMSE"],
    ).reset_index(drop=True)

    # ── 8. Export ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Exporting models and metrics")
    print("=" * 60)

    CONFIG["feature_columns"] = list(feature_columns)
    export_models_and_metrics(
        model_entries=model_entries,
        ensemble_definition=ensemble_definition,
        config=CONFIG,
        one_step_metrics_df=one_step_metrics_df,
        recursive_metrics_df=recursive_metrics_df,
        latest_metrics_path=Path("outputs/metrics/latest_metrics.json"),
        models_dir=Path("models"),
    )
    print(f"  Models → {Path('models').resolve()}")
    print(f"  Metrics → {Path('outputs/metrics/latest_metrics.json').resolve()}")

    # ── 9. Plots ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Generating plots")
    print("=" * 60)

    ranked_models = one_step_metrics_df["Model"].tolist()
    one_step_plot_df = pd.DataFrame({"timestamp": test_df["timestamp"]})
    for model_name in ranked_models:
        one_step_plot_df[model_name] = one_step_predictions[model_name]

    plot_model_comparison(
        test_df=test_df,
        prediction_frame=one_step_plot_df[["timestamp"] + ranked_models],
        title="One-Step Model Comparison on the Test Period",
    )

    best_recursive_model = (
        recursive_metrics_df.loc[recursive_metrics_df["Scope"] == "Daylight only"]
        .sort_values("RMSE")
        .iloc[0]["Model"]
    )
    plot_recursive_backtest(
        backtest_df=recursive_backtests[best_recursive_model],
        model_label=best_recursive_model,
    )

    # ── 10. Future Forecast ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10: Generating 24-hour future forecast")
    print("=" * 60)

    best_model_name = best_recursive_model
    print(f"  Best model: {best_model_name}")

    history_features = feature_df_full.dropna(
        subset=feature_columns + ["power_kw"],
    ).copy()
    future_weather = get_future_weather(
        latitude=CONFIG["latitude"],
        longitude=CONFIG["longitude"],
        last_timestamp=history_features["timestamp"].max(),
        horizon_steps=FORECAST_STEPS,
        timezone=CONFIG["timezone"],
        historical_weather_5min=weather_5min,
    )

    if ensemble_definition and best_model_name == ensemble_definition.get("name"):
        member_outputs = {}
        for member_name in ensemble_definition["members"]:
            member_forecast = recursive_forecast(
                model_entry=model_entries[member_name],
                history_features=history_features,
                future_weather=future_weather,
                feature_columns=feature_columns,
                sequence_length=CONFIG["sequence_length"],
            )
            member_forecast["forecast_power_kw"] = apply_affine_calibrator(
                member_forecast["forecast_power_kw"].to_numpy(),
                model_entries[member_name].get("calibrator"),
            )
            member_outputs[member_name] = member_forecast

        base_member = ensemble_definition["members"][0]
        forecast_df = member_outputs[base_member].loc[
            :, ["timestamp", "ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT", "weather_source"]
        ].copy()
        forecast_df["forecast_power_kw"] = 0.0
        for member_name, weight in ensemble_definition["weights"].items():
            forecast_df["forecast_power_kw"] += (
                float(weight) * member_outputs[member_name]["forecast_power_kw"].to_numpy()
            )
    else:
        forecast_df = recursive_forecast(
            model_entry=model_entries[best_model_name],
            history_features=history_features,
            future_weather=future_weather,
            feature_columns=feature_columns,
            sequence_length=CONFIG["sequence_length"],
        )
        forecast_df["forecast_power_kw"] = apply_affine_calibrator(
            forecast_df["forecast_power_kw"].to_numpy(),
            model_entries[best_model_name].get("calibrator"),
        )

    print(f"  Weather source: {future_weather['weather_source'].iloc[0]}")
    plot_future_forecast(forecast_df)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
