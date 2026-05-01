"""
Feature engineering and data splitting.

Adds cyclic time encodings, lag/rolling statistics, daylight flags,
and sanitises nighttime power anomalies. Also provides chronological
train/val/test splitting utilities.
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append cyclic and calendar time features to the DataFrame."""
    df = frame.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_and_rolling_features(
    frame: pd.DataFrame,
    target_col: str = "power_kw",
) -> pd.DataFrame:
    """Add lag, rolling mean, rolling std, and irradiance rolling mean features."""
    df = frame.copy()
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_288"] = df[target_col].shift(288)
    df["rolling_mean_3"] = df[target_col].shift(1).rolling(3).mean()
    df["rolling_mean_12"] = df[target_col].shift(1).rolling(12).mean()
    df["rolling_mean_24"] = df[target_col].shift(1).rolling(24).mean()
    df["rolling_std_12"] = df[target_col].shift(1).rolling(12).std()
    df["irradiance_rolling_mean_12"] = (
        df["ALLSKY_SFC_SW_DWN"].shift(1).rolling(12).mean()
    )
    return df


def sanitize_generation_profile(
    frame: pd.DataFrame,
    night_power_threshold_kw: float,
    day_irradiance_threshold: float,
    daylight_hour_window: Tuple[float, float],
) -> pd.DataFrame:
    """Zero out anomalous nighttime generation readings."""
    df = frame.copy()
    local_hour = (
        pd.to_datetime(df["timestamp"]).dt.hour
        + pd.to_datetime(df["timestamp"]).dt.minute / 60.0
    )
    df["clock_hour"] = local_hour
    df["night_weather_flag"] = (
        df["ALLSKY_SFC_SW_DWN"] <= day_irradiance_threshold
    ).astype(int)
    df["night_clock_flag"] = (
        (local_hour < daylight_hour_window[0]) | (local_hour > daylight_hour_window[1])
    ).astype(int)
    df["night_anomaly_flag"] = (
        (df["night_weather_flag"] == 1)
        & (df["night_clock_flag"] == 1)
        & (df["power_kw"] > night_power_threshold_kw)
    ).astype(int)
    df.loc[df["night_anomaly_flag"] == 1, "power_kw"] = 0.0
    return df


def engineer_features(
    merged_df: pd.DataFrame,
    night_power_threshold_kw: float,
    day_irradiance_threshold: float,
    daylight_hour_window: Tuple[float, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Full feature engineering pipeline.

    Returns:
        feature_df: Full DataFrame with all features (all hours).
        model_df: Daylight-only subset used for model training.
        feature_columns: List of feature column names for modelling.
    """
    feature_df = sanitize_generation_profile(
        frame=merged_df,
        night_power_threshold_kw=night_power_threshold_kw,
        day_irradiance_threshold=day_irradiance_threshold,
        daylight_hour_window=daylight_hour_window,
    )
    feature_df = add_time_features(feature_df)
    feature_df = add_lag_and_rolling_features(feature_df, target_col="power_kw")
    feature_df["is_daylight"] = (
        feature_df["ALLSKY_SFC_SW_DWN"] > day_irradiance_threshold
    ).astype(int)

    model_df = feature_df.loc[feature_df["is_daylight"] == 1].copy()
    model_df = model_df.dropna().reset_index(drop=True)

    feature_columns = [
        "ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT",
        "hour", "day", "month", "day_of_week", "day_of_year",
        "minute_of_day", "clock_hour",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "lag_1", "lag_24", "lag_288",
        "rolling_mean_3", "rolling_mean_12", "rolling_mean_24",
        "rolling_std_12", "irradiance_rolling_mean_12",
    ]
    return feature_df, model_df, feature_columns


# ── Data splitting ───────────────────────────────────────────────────

def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / val / test."""
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def split_full_frame_by_cutoffs(
    full_frame: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    val_end_ts: pd.Timestamp,
    required_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the full (all-hours) DataFrame using timestamp cutoffs."""
    usable = full_frame.dropna(subset=list(required_columns) + ["power_kw"]).copy()
    usable = usable.sort_values("timestamp").reset_index(drop=True)
    full_train = usable.loc[usable["timestamp"] <= train_end_ts].copy()
    full_val = usable.loc[
        (usable["timestamp"] > train_end_ts) & (usable["timestamp"] <= val_end_ts)
    ].copy()
    full_test = usable.loc[usable["timestamp"] > val_end_ts].copy()
    return full_train, full_val, full_test
