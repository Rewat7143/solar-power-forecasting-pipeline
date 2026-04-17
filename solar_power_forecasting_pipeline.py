# Auto-generated Python mirror of the notebook.
# The notebook remains the primary artifact.

# Notebook setup cell omitted from the Python mirror.

import math
import random
import sys
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)

CONFIG = {
    "data_source_mode": "local_xlsx",  # choose from: local_xlsx, google_sheets
    "workspace_root": Path("."),
    "local_xlsx_glob": "**/Inst_*.xlsx",
    "spreadsheet_id_or_url": None,  # set this for Google Sheets mode
    "google_sheet_tabs": ["MTR_24", "MTR_25", "MTR_26", "MTR_27", "MTR_28", "MTR_29", "Gen_Sum"],
    "google_csv_urls": {
        # Optional direct public CSV links, e.g.
        # "Gen_Sum": "https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=Gen_Sum"
    },
    "meter_tabs": ["MTR_24", "MTR_25", "MTR_26", "MTR_27", "MTR_28", "MTR_29"],
    "summary_tab": "Gen_Sum",
    "date_range": ("2025-07-14", "2026-04-12"),
    "latitude": 17.9374,
    "longitude": 79.5960,
    "timezone": "Asia/Kolkata",
    "nasa_time_standard": "LST",  # LST avoids the 30-minute UTC-to-IST shift artifact for this site
    "target_column": "power_kw",
    "frequency": "5min",
    "night_power_threshold_kw": 0.05,
    "day_irradiance_threshold": 20.0,
    "daylight_hour_window": (5.0, 19.5),
    "sequence_length": 48,  # 48 steps = 4 hours at 5-minute resolution
    "forecast_horizon_hours": 24,
    "backtest_horizon_hours": 24,
    "backtest_stride_hours": 24,
    "random_seed": 42,
    "model_names": [
        "Random Forest",
        "LightGBM",
        "CNN + LSTM",
        "PatchTST",
        "Temporal Fusion Transformer",
        "CNN + Transformer",
    ],
    "rf_param_grid": [
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 18, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.7},
    ],
    "lgbm_param_grid": [
        {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.8},
    ],
    "dl_batch_size": 256,
    "dl_learning_rate": 1e-3,
    "dl_weight_decay": 1e-5,
    "dl_max_epochs": 24,
    "dl_patience": 5,
    "patch_len": 6,
    "patch_stride": 3,
    "transformer_heads": 4,
    "transformer_layers": 2,
    "transformer_d_model": 64,
    "train_fraction": 0.70,
    "val_fraction": 0.15,
}

WEATHER_COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT"]
STEPS_PER_HOUR = int(pd.Timedelta("1h") / pd.Timedelta(CONFIG["frequency"]))
FORECAST_STEPS = CONFIG["forecast_horizon_hours"] * STEPS_PER_HOUR


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

LIGHTGBM_ENABLED = HAS_LIGHTGBM and sys.version_info < (3, 14)
if HAS_LIGHTGBM and not LIGHTGBM_ENABLED:
    print("LightGBM is installed but disabled on Python 3.14 because it can crash the notebook kernel in this environment.")

def extract_spreadsheet_id(spreadsheet_id_or_url: Optional[str]) -> Optional[str]:
    if spreadsheet_id_or_url is None:
        return None
    raw = str(spreadsheet_id_or_url).strip()
    if "/spreadsheets/d/" in raw:
        return raw.split("/spreadsheets/d/")[1].split("/")[0]
    if "drive.google.com/drive/folders/" in raw:
        raise ValueError(
            "A Google Drive folder link does not expose a sheet CSV export endpoint. "
            "Please provide a spreadsheet URL/ID or direct CSV URLs for the required tabs."
        )
    return raw


def build_google_sheet_csv_url(spreadsheet_id: str, sheet_name: str) -> str:
    safe_sheet = quote(sheet_name, safe="")
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={safe_sheet}"


def clean_timeseries_sheet(
    frame: pd.DataFrame,
    power_column: str,
    rename_to: str = "power_kw",
) -> pd.DataFrame:
    if "Timestamp" not in frame.columns or power_column not in frame.columns:
        return pd.DataFrame(columns=["timestamp", rename_to])

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["Timestamp"], errors="coerce")
    cleaned[rename_to] = pd.to_numeric(cleaned[power_column], errors="coerce")
    cleaned = cleaned.loc[cleaned["timestamp"].notna(), ["timestamp", rename_to]].copy()
    cleaned = cleaned.dropna(subset=[rename_to])
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return cleaned.reset_index(drop=True)


def fetch_public_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def load_google_solar_data(
    spreadsheet_id_or_url: Optional[str],
    meter_tabs: Sequence[str],
    summary_tab: str,
    csv_url_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    csv_url_map = csv_url_map or {}
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_id_or_url)

    def get_tab_df(sheet_name: str) -> pd.DataFrame:
        if sheet_name in csv_url_map:
            return fetch_public_csv(csv_url_map[sheet_name])
        if spreadsheet_id is None:
            raise ValueError(
                f"No CSV URL provided for sheet '{sheet_name}' and no spreadsheet ID was supplied."
            )
        return fetch_public_csv(build_google_sheet_csv_url(spreadsheet_id, sheet_name))

    try:
        summary_df = get_tab_df(summary_tab)
        summary_clean = clean_timeseries_sheet(summary_df, power_column="Solar Total kW", rename_to="power_kw")
        if not summary_clean.empty:
            summary_clean["source"] = "google_sheet_summary"
            return summary_clean
    except Exception as exc:
        print(f"Summary tab fallback triggered: {exc}")

    meter_frames = []
    for tab_name in meter_tabs:
        meter_df = get_tab_df(tab_name)
        meter_clean = clean_timeseries_sheet(meter_df, power_column="kW_Total", rename_to=f"{tab_name}_power_kw")
        if not meter_clean.empty:
            meter_frames.append(meter_clean)

    if not meter_frames:
        raise ValueError("No valid solar generation data could be loaded from Google Sheets.")

    merged = meter_frames[0]
    for frame in meter_frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")

    meter_columns = [col for col in merged.columns if col.endswith("_power_kw")]
    merged["power_kw"] = merged[meter_columns].sum(axis=1, min_count=1)
    merged = merged.loc[:, ["timestamp", "power_kw"]].dropna().sort_values("timestamp")
    merged["source"] = "google_sheet_meter_sum"
    return merged.reset_index(drop=True)


def load_solar_from_single_workbook(
    workbook_path: Path,
    meter_tabs: Sequence[str],
    summary_tab: str,
) -> pd.DataFrame:
    excel_file = pd.ExcelFile(workbook_path)

    if summary_tab in excel_file.sheet_names:
        summary_df = excel_file.parse(summary_tab)
        summary_clean = clean_timeseries_sheet(summary_df, power_column="Solar Total kW", rename_to="power_kw")
        if not summary_clean.empty:
            summary_clean["source_file"] = workbook_path.name
            summary_clean["source"] = "local_summary"
            return summary_clean

    meter_frames = []
    for tab_name in meter_tabs:
        if tab_name not in excel_file.sheet_names:
            continue
        meter_df = excel_file.parse(tab_name)
        meter_clean = clean_timeseries_sheet(meter_df, power_column="kW_Total", rename_to=f"{tab_name}_power_kw")
        if not meter_clean.empty:
            meter_frames.append(meter_clean)

    if not meter_frames:
        return pd.DataFrame(columns=["timestamp", "power_kw", "source_file", "source"])

    merged = meter_frames[0]
    for frame in meter_frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")

    meter_columns = [col for col in merged.columns if col.endswith("_power_kw")]
    merged["power_kw"] = merged[meter_columns].sum(axis=1, min_count=1)
    merged = merged.loc[:, ["timestamp", "power_kw"]].dropna().sort_values("timestamp")
    merged["source_file"] = workbook_path.name
    merged["source"] = "local_meter_sum"
    return merged.reset_index(drop=True)


def finalize_solar_dataframe(
    solar_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str,
) -> pd.DataFrame:
    if solar_df.empty:
        raise ValueError("The solar dataframe is empty after ingestion.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(frequency)
    solar_df = solar_df.copy()
    solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"])
    solar_df["power_kw"] = pd.to_numeric(solar_df["power_kw"], errors="coerce")
    solar_df = solar_df.dropna(subset=["timestamp", "power_kw"])
    solar_df = solar_df[(solar_df["timestamp"] >= start_ts) & (solar_df["timestamp"] <= end_ts)]
    solar_df = solar_df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    full_index = pd.date_range(start=solar_df["timestamp"].min(), end=solar_df["timestamp"].max(), freq=frequency)
    solar_df = solar_df.set_index("timestamp").reindex(full_index)
    solar_df.index.name = "timestamp"
    solar_df["missing_power_flag"] = solar_df["power_kw"].isna().astype(int)
    solar_df["power_kw"] = solar_df["power_kw"].interpolate(method="time", limit=2, limit_direction="both")
    solar_df["power_kw"] = solar_df["power_kw"].clip(lower=0)
    metadata_columns = [col for col in solar_df.columns if col not in {"power_kw", "missing_power_flag"}]
    for column in metadata_columns:
        solar_df[column] = solar_df[column].ffill().bfill()
    solar_df = solar_df.dropna(subset=["power_kw"]).reset_index()
    return solar_df


def load_solar_dataset(config: Dict) -> pd.DataFrame:
    mode = config["data_source_mode"]
    if mode == "google_sheets":
        solar_df = load_google_solar_data(
            spreadsheet_id_or_url=config["spreadsheet_id_or_url"],
            meter_tabs=config["meter_tabs"],
            summary_tab=config["summary_tab"],
            csv_url_map=config["google_csv_urls"],
        )
        return finalize_solar_dataframe(
            solar_df=solar_df,
            start_date=config["date_range"][0],
            end_date=config["date_range"][1],
            frequency=config["frequency"],
        )

    if mode == "local_xlsx":
        root = Path(config["workspace_root"])
        workbook_paths = sorted(root.glob(config["local_xlsx_glob"]))
        if not workbook_paths:
            raise FileNotFoundError("No local Excel workbooks matched the configured glob pattern.")

        frames = []
        for workbook_path in workbook_paths:
            frame = load_solar_from_single_workbook(
                workbook_path=workbook_path,
                meter_tabs=config["meter_tabs"],
                summary_tab=config["summary_tab"],
            )
            if not frame.empty:
                frames.append(frame)

        if not frames:
            raise ValueError("No valid local solar workbooks could be parsed.")

        solar_df = pd.concat(frames, ignore_index=True)
        return finalize_solar_dataframe(
            solar_df=solar_df,
            start_date=config["date_range"][0],
            end_date=config["date_range"][1],
            frequency=config["frequency"],
        )

    raise ValueError(f"Unsupported data_source_mode: {mode}")

def fetch_nasa_power_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    time_standard: str = "LST",
    parameters: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    parameters = list(parameters or WEATHER_COLUMNS)
    time_standard = str(time_standard).upper()
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    request_params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": pd.Timestamp(start_date).strftime("%Y%m%d"),
        "end": pd.Timestamp(end_date).strftime("%Y%m%d"),
        "time-standard": time_standard,
    }

    try:
        response = requests.get(
            base_url,
            params={**request_params, "format": "CSV", "header": "false"},
            timeout=120,
        )
        response.raise_for_status()
        weather = pd.read_csv(StringIO(response.text))
        required_cols = {"YEAR", "MO", "DY", "HR"}
        if not required_cols.issubset(weather.columns):
            raise ValueError("NASA CSV response did not contain the expected time columns.")

        weather = weather.rename(columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"})
        weather["timestamp"] = pd.to_datetime(weather[["year", "month", "day", "hour"]])
        if time_standard == "UTC":
            weather["timestamp"] = (
                weather["timestamp"]
                .dt.tz_localize("UTC")
                .dt.tz_convert(timezone)
                .dt.tz_localize(None)
                .add(pd.Timedelta(minutes=30))
                .dt.floor("h")
            )
        weather = weather.drop(columns=["year", "month", "day", "hour"])
    except Exception as csv_error:
        print(f"CSV endpoint fallback triggered: {csv_error}")
        response = requests.get(
            base_url,
            params={**request_params, "format": "JSON"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        parameter_block = payload["properties"]["parameter"]
        timestamp_keys = sorted(next(iter(parameter_block.values())).keys())
        base_timestamp = pd.to_datetime(timestamp_keys, format="%Y%m%d%H")
        if time_standard == "UTC":
            base_timestamp = (
                base_timestamp
                .tz_localize("UTC")
                .tz_convert(timezone)
                .tz_localize(None)
                .add(pd.Timedelta(minutes=30))
                .floor("h")
            )
        weather = pd.DataFrame(
            {
                "timestamp": base_timestamp
            }
        )
        for parameter in parameters:
            weather[parameter] = [parameter_block[parameter].get(key, np.nan) for key in timestamp_keys]

    weather = weather.loc[:, ["timestamp"] + parameters].copy()
    weather = weather.set_index("timestamp").sort_index()
    for column in parameters:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
        weather[column] = weather[column].replace(-999, np.nan)

    weather = weather.interpolate(method="time", limit_direction="both")
    weather = weather.ffill().bfill()
    weather = weather[~weather.index.duplicated(keep="last")]
    weather.index.name = "timestamp"
    return weather.reset_index()


def align_weather_to_index(
    weather_hourly: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    aligned = weather_hourly.copy().set_index("timestamp").sort_index()
    aligned = aligned.reindex(aligned.index.union(target_index)).sort_index()
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].interpolate(method="time", limit_direction="both")
    aligned = aligned.reindex(target_index)
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].ffill().bfill()
    aligned.index.name = "timestamp"
    return aligned.reset_index()


def merge_solar_and_weather(solar_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    merged = solar_df.merge(weather_df, on="timestamp", how="left").sort_values("timestamp")
    for column in WEATHER_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = merged[column].interpolate(method="linear", limit_direction="both")
        merged[column] = merged[column].ffill().bfill()

    merged["power_kw"] = pd.to_numeric(merged["power_kw"], errors="coerce").clip(lower=0)
    merged = merged.dropna(subset=["timestamp", "power_kw"] + WEATHER_COLUMNS)
    return merged.reset_index(drop=True)

def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    df = frame.copy()
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_288"] = df[target_col].shift(288)
    df["rolling_mean_3"] = df[target_col].shift(1).rolling(3).mean()
    df["rolling_mean_12"] = df[target_col].shift(1).rolling(12).mean()
    df["rolling_mean_24"] = df[target_col].shift(1).rolling(24).mean()
    df["rolling_std_12"] = df[target_col].shift(1).rolling(12).std()
    df["irradiance_rolling_mean_12"] = df["ALLSKY_SFC_SW_DWN"].shift(1).rolling(12).mean()
    return df


def sanitize_generation_profile(
    frame: pd.DataFrame,
    night_power_threshold_kw: float,
    day_irradiance_threshold: float,
    daylight_hour_window: Tuple[float, float],
) -> pd.DataFrame:
    df = frame.copy()
    local_hour = pd.to_datetime(df["timestamp"]).dt.hour + pd.to_datetime(df["timestamp"]).dt.minute / 60.0
    df["clock_hour"] = local_hour
    df["night_weather_flag"] = (df["ALLSKY_SFC_SW_DWN"] <= day_irradiance_threshold).astype(int)
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
    feature_df = sanitize_generation_profile(
        frame=merged_df,
        night_power_threshold_kw=night_power_threshold_kw,
        day_irradiance_threshold=day_irradiance_threshold,
        daylight_hour_window=daylight_hour_window,
    )
    feature_df = add_time_features(feature_df)
    feature_df = add_lag_and_rolling_features(feature_df, target_col="power_kw")
    feature_df["is_daylight"] = (feature_df["ALLSKY_SFC_SW_DWN"] > day_irradiance_threshold).astype(int)

    model_df = feature_df.loc[feature_df["is_daylight"] == 1].copy()
    model_df = model_df.dropna().reset_index(drop=True)

    feature_columns = [
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "CLOUD_AMT",
        "hour",
        "day",
        "month",
        "day_of_week",
        "day_of_year",
        "minute_of_day",
        "clock_hour",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "lag_1",
        "lag_24",
        "lag_288",
        "rolling_mean_3",
        "rolling_mean_12",
        "rolling_mean_24",
        "rolling_std_12",
        "irradiance_rolling_mean_12",
    ]
    return feature_df, model_df, feature_columns


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end:val_end].copy()
    test_df = frame.iloc[val_end:].copy()
    return train_df, val_df, test_df


def split_full_frame_by_cutoffs(
    full_frame: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    val_end_ts: pd.Timestamp,
    required_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = full_frame.dropna(subset=list(required_columns) + ["power_kw"]).copy()
    usable = usable.sort_values("timestamp").reset_index(drop=True)
    full_train = usable.loc[usable["timestamp"] <= train_end_ts].copy()
    full_val = usable.loc[(usable["timestamp"] > train_end_ts) & (usable["timestamp"] <= val_end_ts)].copy()
    full_test = usable.loc[usable["timestamp"] > val_end_ts].copy()
    return full_train, full_val, full_test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def fit_affine_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid_mask.sum() < 2:
        return {"slope": 1.0, "intercept": 0.0}

    design = np.column_stack([y_pred[valid_mask], np.ones(valid_mask.sum())])
    slope, intercept = np.linalg.lstsq(design, y_true[valid_mask], rcond=None)[0]
    slope = float(np.clip(slope, 0.0, 3.0))
    intercept = float(intercept)
    return {"slope": slope, "intercept": intercept}


def apply_affine_calibrator(preds: np.ndarray, calibrator: Optional[Dict[str, float]]) -> np.ndarray:
    arr = np.asarray(preds, dtype=float)
    if not calibrator:
        return np.clip(arr, 0.0, None)
    slope = float(calibrator.get("slope", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return np.clip(slope * arr + intercept, 0.0, None)

@dataclass
class SequenceBundle:
    X_all_scaled: np.ndarray
    y_all_scaled: np.ndarray
    X_sequences: np.ndarray
    y_sequences: np.ndarray
    sequence_target_positions: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


class WindowedDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, index: int):
        return self.X_seq[index], self.y_seq[index]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        self.positional = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional[:, : x.size(1), :]


class GatedResidualBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = F.elu(self.fc1(x))
        z = self.dropout(self.fc2(z))
        gated = torch.sigmoid(self.gate(x)) * z
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight_net = nn.Linear(n_features, n_features)
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_net(x), dim=-1)
        selected = x * weights
        return self.proj(selected), weights


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :]).squeeze(-1)


class CNNTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.positional = LearnablePositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.positional(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.num_patches = 1 + max(0, (seq_len - patch_len) // stride)
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.positional = LearnablePositionalEncoding(max(self.num_patches, 1), d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = []
        for start in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, start : start + self.patch_len, :].reshape(x.size(0), -1)
            patches.append(patch)
        patch_tensor = torch.stack(patches, dim=1)
        patch_tensor = self.patch_proj(patch_tensor)
        patch_tensor = self.positional(patch_tensor)
        encoded = self.encoder(patch_tensor)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class TemporalFusionTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.variable_selection = VariableSelectionNetwork(n_features, d_model)
        self.pre_grn = GatedResidualBlock(d_model, d_model * 2, dropout=dropout)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_grn = GatedResidualBlock(d_model, d_model * 2, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected, _ = self.variable_selection(x)
        encoded = self.pre_grn(selected)
        lstm_out, _ = self.lstm(encoded)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, need_weights=False)
        fused = self.post_grn(lstm_out + attn_out)
        return self.head(fused[:, -1, :]).squeeze(-1)


def build_scaled_sequences(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str,
    train_size: int,
    val_size: int,
    sequence_length: int,
) -> SequenceBundle:
    full_X = frame[feature_columns].to_numpy(dtype=float)
    full_y = frame[[target_col]].to_numpy(dtype=float)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(full_X[:train_size])
    target_scaler.fit(full_y[:train_size])

    X_all_scaled = feature_scaler.transform(full_X)
    y_all_scaled = target_scaler.transform(full_y).reshape(-1)

    X_seq = []
    y_seq = []
    positions = []
    for end_idx in range(sequence_length - 1, len(frame)):
        start_idx = end_idx - sequence_length + 1
        X_seq.append(X_all_scaled[start_idx : end_idx + 1])
        y_seq.append(y_all_scaled[end_idx])
        positions.append(end_idx)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    positions = np.asarray(positions, dtype=int)

    train_mask = positions < train_size
    val_mask = (positions >= train_size) & (positions < train_size + val_size)
    test_mask = positions >= train_size + val_size

    return SequenceBundle(
        X_all_scaled=X_all_scaled,
        y_all_scaled=y_all_scaled,
        X_sequences=X_seq,
        y_sequences=y_seq,
        sequence_target_positions=positions,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def build_sequence_model(model_name: str, n_features: int, seq_len: int, config: Dict) -> nn.Module:
    if model_name == "CNN + LSTM":
        return CNNLSTMRegressor(n_features=n_features, d_model=config["transformer_d_model"])
    if model_name == "PatchTST":
        return PatchTSTRegressor(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=config["patch_len"],
            stride=config["patch_stride"],
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
            n_layers=config["transformer_layers"],
        )
    if model_name == "Temporal Fusion Transformer":
        return TemporalFusionTransformerRegressor(
            n_features=n_features,
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
        )
    if model_name == "CNN + Transformer":
        return CNNTransformerRegressor(
            n_features=n_features,
            seq_len=seq_len,
            d_model=config["transformer_d_model"],
            n_heads=config["transformer_heads"],
            n_layers=config["transformer_layers"],
        )
    raise ValueError(f"Unsupported sequence model: {model_name}")


def train_sequence_model(
    model: nn.Module,
    bundle: SequenceBundle,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[nn.Module, pd.DataFrame]:
    train_dataset = WindowedDataset(bundle.X_sequences[bundle.train_mask], bundle.y_sequences[bundle.train_mask])
    val_dataset = WindowedDataset(bundle.X_sequences[bundle.val_mask], bundle.y_sequences[bundle.val_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        history_rows.append({"epoch": epoch, "train_loss": mean_train_loss, "val_loss": mean_val_loss})

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows)


def predict_sequence_model(model: nn.Module, X_seq: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)
        preds = model(tensor).detach().cpu().numpy()
    return preds


def time_series_cv_search(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str,
    param_grid: Sequence[Dict],
    random_seed: int,
) -> Tuple[object, pd.DataFrame]:
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X = train_val_df[feature_columns]
    y = train_val_df[target_col]
    tscv = TimeSeriesSplit(n_splits=4)
    cv_rows = []
    best_params = None
    best_rmse = float("inf")

    def build_model(params: Dict):
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=random_seed, n_jobs=-1, **params)
        if model_name == "LightGBM":
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM is not installed. Run the setup cell first.")
            return LGBMRegressor(
                objective="regression",
                random_state=random_seed,
                verbose=-1,
                n_jobs=-1,
                **params,
            )
        raise ValueError(f"Unsupported tree model: {model_name}")

    for params in param_grid:
        fold_rmses = []
        for fold_id, (fit_idx, hold_idx) in enumerate(tscv.split(X), start=1):
            model = build_model(params)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            fold_pred = model.predict(X.iloc[hold_idx])
            try:
                fold_rmse = mean_squared_error(y.iloc[hold_idx], fold_pred, squared=False)
            except TypeError:
                fold_rmse = np.sqrt(mean_squared_error(y.iloc[hold_idx], fold_pred))
            fold_rmses.append(fold_rmse)
            cv_rows.append({"model": model_name, "fold": fold_id, "params": params, "rmse": fold_rmse})

        avg_rmse = float(np.mean(fold_rmses))
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    best_model = build_model(best_params)
    best_model.fit(X, y)
    return best_model, pd.DataFrame(cv_rows)

def compute_feature_row_from_history(
    history_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    weather_row: pd.Series,
) -> Dict[str, float]:
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

    row.update(
        {
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
        }
    )

    row["lag_1"] = float(history_power.iloc[-1]) if len(history_power) >= 1 else np.nan
    row["lag_24"] = float(history_power.iloc[-24]) if len(history_power) >= 24 else np.nan
    row["lag_288"] = float(history_power.iloc[-288]) if len(history_power) >= 288 else np.nan
    row["rolling_mean_3"] = float(history_power.iloc[-3:].mean()) if len(history_power) >= 3 else np.nan
    row["rolling_mean_12"] = float(history_power.iloc[-12:].mean()) if len(history_power) >= 12 else np.nan
    row["rolling_mean_24"] = float(history_power.iloc[-24:].mean()) if len(history_power) >= 24 else np.nan
    row["rolling_std_12"] = float(history_power.iloc[-12:].std()) if len(history_power) >= 12 else np.nan

    recent_irr = history_df["ALLSKY_SFC_SW_DWN"].astype(float)
    row["irradiance_rolling_mean_12"] = float(recent_irr.iloc[-12:].mean()) if len(recent_irr) >= 12 else np.nan
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
    history = historical_weather_5min.copy().set_index("timestamp").sort_index()
    history = history[WEATHER_COLUMNS].copy()
    history_df = history.reset_index()
    history_df["slot"] = history_df["timestamp"].dt.hour * STEPS_PER_HOUR + history_df["timestamp"].dt.minute // 5
    recent_window = history_df.loc[history_df["timestamp"] >= history_df["timestamp"].max() - pd.Timedelta(days=30)].copy()

    slot_profile = recent_window.groupby("slot")[["T2M", "CLOUD_AMT"]].median()
    irradiance_profile = recent_window.groupby("slot")["ALLSKY_SFC_SW_DWN"].apply(
        lambda s: float(s[s > CONFIG["day_irradiance_threshold"]].median()) if (s > CONFIG["day_irradiance_threshold"]).any() else float(s.median())
    )
    full_irradiance_profile = history_df.groupby("slot")["ALLSKY_SFC_SW_DWN"].apply(
        lambda s: float(s[s > CONFIG["day_irradiance_threshold"]].median()) if (s > CONFIG["day_irradiance_threshold"]).any() else float(s.median())
    )
    slot_profile["ALLSKY_SFC_SW_DWN"] = irradiance_profile
    slot_profile["ALLSKY_SFC_SW_DWN"] = slot_profile["ALLSKY_SFC_SW_DWN"].fillna(full_irradiance_profile)
    slot_profile = slot_profile.reset_index()

    future = pd.DataFrame({"timestamp": future_index})
    future["slot"] = future["timestamp"].dt.hour * STEPS_PER_HOUR + future["timestamp"].dt.minute // 5
    future = future.merge(slot_profile, on="slot", how="left")

    previous_day = history.reindex(future["timestamp"] - pd.Timedelta(days=1))
    if not previous_day.empty:
        previous_day = previous_day.reset_index(drop=True)
        for column in WEATHER_COLUMNS:
            future[column] = np.where(previous_day[column].notna(), previous_day[column].to_numpy(), future[column])
            future[column] = pd.to_numeric(pd.Series(future[column]).ffill().bfill(), errors="coerce")

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
    history = history_features.copy().sort_values("timestamp").reset_index(drop=True)
    forecasts = []

    for _, weather_row in future_weather.iterrows():
        feature_row = compute_feature_row_from_history(history, pd.Timestamp(weather_row["timestamp"]), weather_row)

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
            x_tensor = torch.tensor(scaled_candidate, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model_entry["model"](x_tensor).cpu().numpy().reshape(-1, 1)
            pred_kw = float(model_entry["target_scaler"].inverse_transform(pred_scaled)[0, 0])

        pred_kw = float(np.clip(pred_kw, 0.0, None))
        feature_row["power_kw"] = pred_kw
        forecasts.append(
            {
                "timestamp": pd.Timestamp(feature_row["timestamp"]),
                "forecast_power_kw": pred_kw,
                "ALLSKY_SFC_SW_DWN": feature_row["ALLSKY_SFC_SW_DWN"],
                "T2M": feature_row["T2M"],
                "CLOUD_AMT": feature_row["CLOUD_AMT"],
                "weather_source": weather_row.get("weather_source", "unknown"),
            }
        )
        history = pd.concat([history, pd.DataFrame([feature_row])], ignore_index=True)

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


def run_recursive_backtest(
    model_entry: Dict[str, object],
    full_history_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
    horizon_steps: int,
    stride_steps: int,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    evaluation_df = evaluation_df.sort_values("timestamp").reset_index(drop=True)
    rows = []

    for start_idx in range(0, max(len(evaluation_df) - horizon_steps + 1, 0), stride_steps):
        future_window = evaluation_df.iloc[start_idx : start_idx + horizon_steps].copy()
        if len(future_window) < horizon_steps:
            continue

        history_cutoff = future_window["timestamp"].iloc[0] - pd.Timedelta(CONFIG["frequency"])
        history_slice = full_history_df.loc[full_history_df["timestamp"] <= history_cutoff].copy()
        history_slice = history_slice.dropna(subset=list(feature_columns) + ["power_kw"]).sort_values("timestamp")
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
        raise ValueError("Recursive backtest could not be created. Check horizon length and available history.")

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


def plot_recursive_backtest(backtest_df: pd.DataFrame, model_label: str) -> None:
    plot_df = backtest_df.loc[backtest_df["is_daylight"] == 1].sort_values("timestamp").copy()
    plt.figure(figsize=(16, 6))
    plt.plot(plot_df["timestamp"], plot_df["actual_power_kw"], label="Actual", linewidth=2, color="#1f2937")
    plt.plot(
        plot_df["timestamp"],
        plot_df["forecast_power_kw"],
        label=f"{model_label} recursive backtest",
        linewidth=1.6,
        color="#0f766e",
    )
    plt.title(f"Rolling 24-Hour Recursive Backtest: {model_label}")
    plt.xlabel("Timestamp")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    test_df: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    title: str,
    top_k: int = 4,
) -> None:
    top_columns = [col for col in prediction_frame.columns if col != "timestamp"][:top_k]
    plt.figure(figsize=(16, 6))
    plt.plot(test_df["timestamp"], test_df["power_kw"], label="Actual", linewidth=2, color="#1f2937")
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0f766e"]
    for idx, column in enumerate(top_columns):
        plt.plot(
            prediction_frame["timestamp"],
            prediction_frame[column],
            label=column,
            alpha=0.85,
            color=palette[idx % len(palette)],
        )
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_future_forecast(forecast_df: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 5))
    plt.plot(
        forecast_df["timestamp"],
        forecast_df["forecast_power_kw"],
        linewidth=2.5,
        color="#059669",
        label="Forecasted solar power",
    )
    plt.fill_between(
        forecast_df["timestamp"],
        forecast_df["forecast_power_kw"],
        alpha=0.20,
        color="#10b981",
    )
    plt.title("Next 24 Hours Solar Power Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Forecast Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

solar_df = load_solar_dataset(CONFIG)
print("Solar rows:", len(solar_df))
print("Solar time span:", solar_df["timestamp"].min(), "to", solar_df["timestamp"].max())
print("Observed max power (kW):", round(solar_df["power_kw"].max(), 3))
solar_df.head()

weather_hourly = fetch_nasa_power_hourly(
    latitude=CONFIG["latitude"],
    longitude=CONFIG["longitude"],
    start_date=CONFIG["date_range"][0],
    end_date=CONFIG["date_range"][1],
    timezone=CONFIG["timezone"],
    time_standard=CONFIG["nasa_time_standard"],
    parameters=WEATHER_COLUMNS,
)

weather_5min = align_weather_to_index(weather_hourly, pd.DatetimeIndex(solar_df["timestamp"]))
merged_df = merge_solar_and_weather(solar_df, weather_5min)

print("Merged rows:", len(merged_df))
print("Missing values after merge:")
print(merged_df.isna().sum())
merged_df.head()

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

print("Full merged rows:", len(merged_df))
print("Modeling rows after daylight filter:", len(model_df))
print("Train / Val / Test sizes:", len(train_df), len(val_df), len(test_df))
print("Night anomaly rows zeroed during cleaning:", int(feature_df_full["night_anomaly_flag"].sum()))
print("Full backtest rows:", len(full_test_df))
print("Feature columns:", feature_columns)
model_df.head()

model_entries = {}
one_step_predictions = {}
one_step_rows = []
cv_result_frames = []

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
        "name": model_name,
        "family": "tree",
        "model": model,
        "calibrator": calibrator,
    }
    one_step_predictions[model_name] = preds
    one_step_rows.append({"Model": model_name, **metrics})
    cv_result_frames.append(cv_results)
    print(f"{model_name} one-step metrics:", metrics)

tree_cv_results_df = pd.concat(cv_result_frames, ignore_index=True)
tree_cv_results_df.head()

sequence_bundle = build_scaled_sequences(
    frame=model_df,
    feature_columns=feature_columns,
    target_col="power_kw",
    train_size=len(train_df),
    val_size=len(val_df),
    sequence_length=CONFIG["sequence_length"],
)

sequence_training_histories = {}
sequence_model_names = ["CNN + LSTM", "PatchTST", "Temporal Fusion Transformer", "CNN + Transformer"]

for model_name in sequence_model_names:
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

    val_preds_scaled = predict_sequence_model(seq_model, sequence_bundle.X_sequences[sequence_bundle.val_mask])
    val_preds_raw = sequence_bundle.target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(-1)
    calibrator = fit_affine_calibrator(val_df["power_kw"].to_numpy(), val_preds_raw)

    preds_scaled = predict_sequence_model(seq_model, sequence_bundle.X_sequences[sequence_bundle.test_mask])
    preds_raw = sequence_bundle.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    preds = apply_affine_calibrator(preds_raw, calibrator)
    metrics = regression_metrics(test_df["power_kw"].to_numpy(), preds)

    model_entries[model_name] = {
        "name": model_name,
        "family": "sequence",
        "model": seq_model,
        "feature_scaler": sequence_bundle.feature_scaler,
        "target_scaler": sequence_bundle.target_scaler,
        "calibrator": calibrator,
    }
    one_step_predictions[model_name] = preds
    one_step_rows.append({"Model": model_name, **metrics})
    sequence_training_histories[model_name] = history_df
    print(f"{model_name} one-step metrics:", metrics)

sequence_training_summary_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            "best_val_loss": float(history_df["val_loss"].min()),
            "epochs_trained": int(history_df["epoch"].max()),
        }
        for model_name, history_df in sequence_training_histories.items()
    ]
).sort_values("best_val_loss")

sequence_training_summary_df

one_step_metrics_df = pd.DataFrame(
    one_step_rows
).sort_values("RMSE").reset_index(drop=True)

one_step_metrics_df

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
        backtest_df["forecast_power_kw"].to_numpy(),
        model_entry.get("calibrator"),
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
    recursive_rows.extend(
        [
            {"Model": model_name, "Scope": "All hours", **all_metrics},
            {"Model": model_name, "Scope": "Daylight only", **daylight_metrics},
        ]
    )
    print(f"{model_name} recursive daylight metrics:", daylight_metrics)

base_daylight = pd.DataFrame(recursive_rows)
base_daylight = base_daylight.loc[base_daylight["Scope"] == "Daylight only"].sort_values("RMSE")
top2_models = base_daylight["Model"].head(2).tolist()

if len(top2_models) == 2:
    m1, m2 = top2_models
    rmse_values = base_daylight.set_index("Model").loc[[m1, m2], "RMSE"].to_numpy(dtype=float)
    inv_rmse = 1.0 / np.clip(rmse_values, 1e-6, None)
    weights = inv_rmse / inv_rmse.sum()
    ensemble_name = f"Ensemble ({m1} + {m2})"

    b1 = recursive_backtests[m1].loc[:, ["timestamp", "forecast_origin", "actual_power_kw", "is_daylight", "forecast_power_kw"]].copy()
    b2 = recursive_backtests[m2].loc[:, ["timestamp", "forecast_origin", "forecast_power_kw"]].copy()
    b1 = b1.rename(columns={"forecast_power_kw": "pred_1"})
    b2 = b2.rename(columns={"forecast_power_kw": "pred_2"})
    ensemble_backtest = b1.merge(b2, on=["timestamp", "forecast_origin"], how="inner")
    ensemble_backtest["forecast_power_kw"] = (
        float(weights[0]) * ensemble_backtest["pred_1"]
        + float(weights[1]) * ensemble_backtest["pred_2"]
    )
    ensemble_backtest = ensemble_backtest.loc[:, ["timestamp", "forecast_origin", "actual_power_kw", "is_daylight", "forecast_power_kw"]]

    ens_all_metrics = regression_metrics(
        ensemble_backtest["actual_power_kw"].to_numpy(),
        ensemble_backtest["forecast_power_kw"].to_numpy(),
    )
    ens_daylight_df = ensemble_backtest.loc[ensemble_backtest["is_daylight"] == 1].copy()
    ens_daylight_metrics = regression_metrics(
        ens_daylight_df["actual_power_kw"].to_numpy(),
        ens_daylight_df["forecast_power_kw"].to_numpy(),
    )

    recursive_backtests[ensemble_name] = ensemble_backtest
    recursive_rows.extend(
        [
            {"Model": ensemble_name, "Scope": "All hours", **ens_all_metrics},
            {"Model": ensemble_name, "Scope": "Daylight only", **ens_daylight_metrics},
        ]
    )
    ensemble_definition = {
        "name": ensemble_name,
        "members": [m1, m2],
        "weights": {m1: float(weights[0]), m2: float(weights[1])},
    }
    print(f"{ensemble_name} recursive daylight metrics:", ens_daylight_metrics)
    print("Ensemble weights:", ensemble_definition["weights"])

recursive_metrics_df = pd.DataFrame(recursive_rows).sort_values(["Scope", "RMSE"]).reset_index(drop=True)

recursive_metrics_df

ranked_models = one_step_metrics_df["Model"].tolist()
one_step_plot_df = pd.DataFrame({"timestamp": test_df["timestamp"]})
for model_name in ranked_models:
    one_step_plot_df[model_name] = one_step_predictions[model_name]

plot_model_comparison(
    test_df=test_df,
    prediction_frame=one_step_plot_df[["timestamp"] + ranked_models],
    title="One-Step Model Comparison on the Test Period",
)

best_recursive_model_name = (
    recursive_metrics_df.loc[recursive_metrics_df["Scope"] == "Daylight only"]
    .sort_values("RMSE")
    .iloc[0]["Model"]
)

plot_recursive_backtest(
    backtest_df=recursive_backtests[best_recursive_model_name],
    model_label=best_recursive_model_name,
)

best_model_name = best_recursive_model_name
print("Best model selected for recursive forecasting:", best_model_name)

history_features = feature_df_full.dropna(subset=feature_columns + ["power_kw"]).copy()
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
    forecast_df = member_outputs[base_member].loc[:, ["timestamp", "ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT", "weather_source"]].copy()
    forecast_df["forecast_power_kw"] = 0.0
    for member_name, weight in ensemble_definition["weights"].items():
        forecast_df["forecast_power_kw"] += float(weight) * member_outputs[member_name]["forecast_power_kw"].to_numpy()
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

print("Future weather source:", future_weather["weather_source"].iloc[0])
print("Future irradiance summary:", future_weather["ALLSKY_SFC_SW_DWN"].describe()[["min", "max", "mean"]].to_dict())
forecast_df.head()

plot_future_forecast(forecast_df)
forecast_df.tail()

