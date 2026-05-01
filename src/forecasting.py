"""
Recursive forecasting and backtesting.

Implements step-by-step autoregressive prediction by building feature
rows from history, fetching or proxying future weather, and evaluating
forecasts against held-out actuals.
"""

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import CONFIG, DEVICE, FORECAST_STEPS, STEPS_PER_HOUR, WEATHER_COLUMNS
from src.evaluation import apply_affine_calibrator, regression_metrics
from src.weather import align_weather_to_index, fetch_nasa_power_hourly


def compute_feature_row_from_history(
    history_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    weather_row: pd.Series,
) -> Dict[str, float]:
    """Build a single feature row for the next forecast step using historical context."""
    history_power = history_df["power_kw"].astype(float)
    row = {
        "timestamp": pd.Timestamp(timestamp),
        "power_kw": np.nan,
        "ALLSKY_SFC_SW_DWN": float(weather_row["ALLSKY_SFC_SW_DWN"]),
        "T2M": float(weather_row["T2M"]),
        "CLOUD_AMT": float(weather_row["CLOUD_AMT"]),
    }

    hour = timestamp.hour + timestamp.minute / 60.0
    day = timestamp.day
    month = timestamp.month
    day_of_week = timestamp.dayofweek
    day_of_year = timestamp.dayofyear
    minute_of_day = timestamp.hour * 60 + timestamp.minute

    row.update({
        "hour": hour,
        "clock_hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "minute_of_day": minute_of_day,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "month_sin": math.sin(2 * math.pi * month / 12.0),
        "month_cos": math.cos(2 * math.pi * month / 12.0),
        "dow_sin": math.sin(2 * math.pi * day_of_week / 7.0),
        "dow_cos": math.cos(2 * math.pi * day_of_week / 7.0),
        "doy_sin": math.sin(2 * math.pi * day_of_year / 365.25),
        "doy_cos": math.cos(2 * math.pi * day_of_year / 365.25),
    })

    row["lag_1"] = float(history_power.iloc[-1]) if len(history_power) >= 1 else np.nan
    row["lag_24"] = float(history_power.iloc[-24]) if len(history_power) >= 24 else np.nan
    row["lag_288"] = float(history_power.iloc[-288]) if len(history_power) >= 288 else np.nan
    row["rolling_mean_3"] = float(history_power.iloc[-3:].mean()) if len(history_power) >= 3 else np.nan
    row["rolling_mean_12"] = float(history_power.iloc[-12:].mean()) if len(history_power) >= 12 else np.nan
    row["rolling_mean_24"] = float(history_power.iloc[-24:].mean()) if len(history_power) >= 24 else np.nan
    row["rolling_std_12"] = float(history_power.iloc[-12:].std()) if len(history_power) >= 12 else np.nan

    recent_irr = history_df["ALLSKY_SFC_SW_DWN"].astype(float)
    row["irradiance_rolling_mean_12"] = (
        float(recent_irr.iloc[-12:].mean()) if len(recent_irr) >= 12 else np.nan
    )
    row["night_weather_flag"] = int(row["ALLSKY_SFC_SW_DWN"] <= CONFIG["day_irradiance_threshold"])
    row["night_clock_flag"] = int(
        (row["clock_hour"] < CONFIG["daylight_hour_window"][0])
        or (row["clock_hour"] > CONFIG["daylight_hour_window"][1])
    )
    row["night_anomaly_flag"] = 0
    row["is_daylight"] = int(
        (row["ALLSKY_SFC_SW_DWN"] > CONFIG["day_irradiance_threshold"])
        and (row["night_clock_flag"] == 0)
    )
    return row


def build_future_weather_proxy(
    historical_weather_5min: pd.DataFrame,
    future_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a proxy weather forecast from recent historical profiles."""
    history = historical_weather_5min.copy().set_index("timestamp").sort_index()
    history = history[WEATHER_COLUMNS].copy()
    history_df = history.reset_index()
    history_df["slot"] = (
        history_df["timestamp"].dt.hour * STEPS_PER_HOUR
        + history_df["timestamp"].dt.minute // 5
    )
    recent_window = history_df.loc[
        history_df["timestamp"] >= history_df["timestamp"].max() - pd.Timedelta(days=30)
    ].copy()

    slot_profile = recent_window.groupby("slot")[["T2M", "CLOUD_AMT"]].median()
    irradiance_profile = recent_window.groupby("slot")["ALLSKY_SFC_SW_DWN"].apply(
        lambda s: (
            float(s[s > CONFIG["day_irradiance_threshold"]].median())
            if (s > CONFIG["day_irradiance_threshold"]).any()
            else float(s.median())
        )
    )
    full_irradiance_profile = history_df.groupby("slot")["ALLSKY_SFC_SW_DWN"].apply(
        lambda s: (
            float(s[s > CONFIG["day_irradiance_threshold"]].median())
            if (s > CONFIG["day_irradiance_threshold"]).any()
            else float(s.median())
        )
    )
    slot_profile["ALLSKY_SFC_SW_DWN"] = irradiance_profile
    slot_profile["ALLSKY_SFC_SW_DWN"] = slot_profile["ALLSKY_SFC_SW_DWN"].fillna(
        full_irradiance_profile,
    )
    slot_profile = slot_profile.reset_index()

    future = pd.DataFrame({"timestamp": future_index})
    future["slot"] = (
        future["timestamp"].dt.hour * STEPS_PER_HOUR + future["timestamp"].dt.minute // 5
    )
    future = future.merge(slot_profile, on="slot", how="left")

    previous_day = history.reindex(future["timestamp"] - pd.Timedelta(days=1))
    if not previous_day.empty:
        previous_day = previous_day.reset_index(drop=True)
        for column in WEATHER_COLUMNS:
            future[column] = np.where(
                previous_day[column].notna(), previous_day[column].to_numpy(), future[column],
            )
            future[column] = pd.to_numeric(
                pd.Series(future[column]).ffill().bfill(), errors="coerce",
            )

    future["ALLSKY_SFC_SW_DWN"] = future["ALLSKY_SFC_SW_DWN"].clip(lower=0)
    future["CLOUD_AMT"] = future["CLOUD_AMT"].clip(lower=0, upper=100)
    future["clock_hour"] = future["timestamp"].dt.hour + future["timestamp"].dt.minute / 60.0
    future.loc[
        (future["clock_hour"] < CONFIG["daylight_hour_window"][0])
        | (future["clock_hour"] > CONFIG["daylight_hour_window"][1]),
        "ALLSKY_SFC_SW_DWN",
    ] = 0.0
    future = future.drop(columns=["clock_hour"])

    if future["ALLSKY_SFC_SW_DWN"].max() <= CONFIG["day_irradiance_threshold"]:
        fallback_profile = history_df.groupby("slot")["ALLSKY_SFC_SW_DWN"].max()
        future["ALLSKY_SFC_SW_DWN"] = future["slot"].map(fallback_profile).fillna(0.0)
        future["ALLSKY_SFC_SW_DWN"] = future["ALLSKY_SFC_SW_DWN"].clip(lower=0)

    future["weather_source"] = "recent_profile_proxy"
    return future.drop(columns=["slot"])


def get_future_weather(
    latitude: float,
    longitude: float,
    last_timestamp: pd.Timestamp,
    horizon_steps: int,
    timezone: str,
    historical_weather_5min: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch future weather from NASA POWER, falling back to proxy if unavailable."""
    future_index = pd.date_range(
        start=last_timestamp + pd.Timedelta(CONFIG["frequency"]),
        periods=horizon_steps,
        freq=CONFIG["frequency"],
    )

    future_start = future_index.min().date().isoformat()
    future_end = future_index.max().date().isoformat()
    future_weather = None

    try:
        nasa_future_hourly = fetch_nasa_power_hourly(
            latitude=latitude,
            longitude=longitude,
            start_date=future_start,
            end_date=future_end,
            timezone=timezone,
            time_standard=CONFIG["nasa_time_standard"],
            parameters=WEATHER_COLUMNS,
        )
        future_weather = align_weather_to_index(nasa_future_hourly, future_index)
        coverage = future_weather[WEATHER_COLUMNS].notna().mean().min()
        if coverage < 0.90:
            future_weather = None
        else:
            future_weather["weather_source"] = "nasa_power"
    except Exception as exc:
        print(f"Future NASA weather fallback triggered: {exc}")

    if future_weather is None:
        future_weather = build_future_weather_proxy(historical_weather_5min, future_index)

    return future_weather


def recursive_forecast(
    model_entry: Dict[str, object],
    history_features: pd.DataFrame,
    future_weather: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
) -> pd.DataFrame:
    """Generate step-by-step autoregressive forecasts over the future weather window."""
    history = history_features.copy().sort_values("timestamp").reset_index(drop=True)
    forecasts = []

    for _, weather_row in future_weather.iterrows():
        feature_row = compute_feature_row_from_history(
            history, pd.Timestamp(weather_row["timestamp"]), weather_row,
        )

        if feature_row["ALLSKY_SFC_SW_DWN"] <= CONFIG["day_irradiance_threshold"]:
            pred_kw = 0.0
        elif model_entry["family"] == "tree":
            row_df = pd.DataFrame([feature_row])
            pred_kw = float(model_entry["model"].predict(row_df[feature_columns])[0])
        else:
            candidate_features = pd.concat(
                [history[feature_columns], pd.DataFrame([feature_row])[feature_columns]],
                ignore_index=True,
            ).tail(sequence_length)
            scaled_candidate = model_entry["feature_scaler"].transform(candidate_features)
            x_tensor = torch.tensor(
                scaled_candidate, dtype=torch.float32, device=DEVICE,
            ).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model_entry["model"](x_tensor).cpu().numpy().reshape(-1, 1)
            pred_kw = float(
                model_entry["target_scaler"].inverse_transform(pred_scaled)[0, 0],
            )

        pred_kw = float(np.clip(pred_kw, 0.0, None))
        feature_row["power_kw"] = pred_kw
        forecasts.append({
            "timestamp": pd.Timestamp(feature_row["timestamp"]),
            "forecast_power_kw": pred_kw,
            "ALLSKY_SFC_SW_DWN": feature_row["ALLSKY_SFC_SW_DWN"],
            "T2M": feature_row["T2M"],
            "CLOUD_AMT": feature_row["CLOUD_AMT"],
            "weather_source": weather_row.get("weather_source", "unknown"),
        })
        history = pd.concat([history, pd.DataFrame([feature_row])], ignore_index=True)

    return pd.DataFrame(forecasts)


def run_recursive_backtest(
    model_entry: Dict[str, object],
    full_history_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
    horizon_steps: int,
    stride_steps: int,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Run rolling-origin recursive backtests and compute metrics."""
    evaluation_df = evaluation_df.sort_values("timestamp").reset_index(drop=True)
    rows = []

    for start_idx in range(0, max(len(evaluation_df) - horizon_steps + 1, 0), stride_steps):
        future_window = evaluation_df.iloc[start_idx : start_idx + horizon_steps].copy()
        if len(future_window) < horizon_steps:
            continue

        history_cutoff = future_window["timestamp"].iloc[0] - pd.Timedelta(CONFIG["frequency"])
        history_slice = full_history_df.loc[full_history_df["timestamp"] <= history_cutoff].copy()
        history_slice = history_slice.dropna(
            subset=list(feature_columns) + ["power_kw"],
        ).sort_values("timestamp")
        if len(history_slice) < max(sequence_length, 288):
            continue

        future_weather = future_window.loc[:, ["timestamp"] + WEATHER_COLUMNS].copy()
        future_weather["weather_source"] = "historical_backtest"

        window_forecast = recursive_forecast(
            model_entry=model_entry,
            history_features=history_slice,
            future_weather=future_weather,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
        )
        window_forecast["actual_power_kw"] = future_window["power_kw"].to_numpy()
        window_forecast["is_daylight"] = future_window["is_daylight"].to_numpy()
        window_forecast["forecast_origin"] = future_window["timestamp"].iloc[0]
        rows.append(window_forecast)

    if not rows:
        raise ValueError(
            "Recursive backtest could not be created. "
            "Check horizon length and available history."
        )

    backtest_df = pd.concat(rows, ignore_index=True)
    all_metrics = regression_metrics(
        backtest_df["actual_power_kw"].to_numpy(),
        backtest_df["forecast_power_kw"].to_numpy(),
    )
    daylight_df = backtest_df.loc[backtest_df["is_daylight"] == 1].copy()
    daylight_metrics = regression_metrics(
        daylight_df["actual_power_kw"].to_numpy(),
        daylight_df["forecast_power_kw"].to_numpy(),
    )
    return backtest_df, all_metrics, daylight_metrics
