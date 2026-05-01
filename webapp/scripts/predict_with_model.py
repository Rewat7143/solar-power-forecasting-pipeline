#!/usr/bin/env python3
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import requests


INTERVAL_MIN = 5
MIN_STD_KW = 2.0
TZ = timezone(timedelta(hours=5, minutes=30))
DAY_IRRADIANCE_THRESHOLD = 20.0
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2]
WEATHER_COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT"]
SOLAR_CACHE_FILE = "solar_dataset_cache.pkl"
WEATHER_CACHE_FILE = "weather_hourly_cache.csv"
WEATHER_CACHE_META_FILE = "weather_hourly_cache_meta.json"
CALIBRATION_FACTOR = 1.1435  # Adjusted calibration factor to match live May 2026 data


def parse_dt(value: str) -> datetime:
    s = (value or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


def read_text(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source) as resp:
            return resp.read().decode("utf-8")
    p = Path(source)
    return p.read_text(encoding="utf-8")


def parse_csv_rows(text: str):
    reader = csv.DictReader(text.splitlines())
    return list(reader)


def slot_of(dt: datetime) -> int:
    return (dt.hour * 60 + dt.minute) // INTERVAL_MIN


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def round2(v: float) -> float:
    return round(float(v), 2)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def clean_timeseries_sheet(frame: pd.DataFrame, power_column: str, rename_to: str = "power_kw") -> pd.DataFrame:
    if "Timestamp" not in frame.columns or power_column not in frame.columns:
        return pd.DataFrame(columns=["timestamp", rename_to])

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["Timestamp"], errors="coerce")
    cleaned[rename_to] = pd.to_numeric(cleaned[power_column], errors="coerce")
    cleaned = cleaned.loc[cleaned["timestamp"].notna(), ["timestamp", rename_to]].copy()
    cleaned = cleaned.dropna(subset=[rename_to])
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return cleaned.reset_index(drop=True)


def load_solar_from_single_workbook(workbook_path: Path) -> pd.DataFrame:
    excel_file = pd.ExcelFile(workbook_path)

    if "Gen_Sum" in excel_file.sheet_names:
        summary_df = excel_file.parse("Gen_Sum")
        summary_clean = clean_timeseries_sheet(summary_df, power_column="Solar Total kW", rename_to="power_kw")
        if not summary_clean.empty:
            summary_clean["source_file"] = workbook_path.name
            summary_clean["source"] = "local_summary"
            return summary_clean

    meter_frames = []
    for tab_name in ["MTR_24", "MTR_25", "MTR_26", "MTR_27", "MTR_28", "MTR_29"]:
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


def load_solar_dataset_local(data_root: Path) -> pd.DataFrame:
    workbook_paths = sorted(data_root.glob("**/Inst_*.xlsx"))
    if not workbook_paths:
        raise FileNotFoundError(f"No local Excel workbooks matched the configured root: {data_root}")

    cache_dir = data_root / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / SOLAR_CACHE_FILE

    latest_workbook_mtime = max(path.stat().st_mtime for path in workbook_paths)
    if cache_path.exists() and cache_path.stat().st_mtime >= latest_workbook_mtime:
        try:
            cached_df = pd.read_pickle(cache_path)
            if not cached_df.empty:
                return cached_df
        except Exception:
            pass

    frames = []
    for workbook_path in workbook_paths:
        try:
            frame = load_solar_from_single_workbook(workbook_path)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue

    if not frames:
        raise ValueError("No valid local solar workbooks could be parsed.")

    solar_df = pd.concat(frames, ignore_index=True)
    solar_df = finalize_solar_dataframe(
        solar_df=solar_df,
        start_date=str(pd.to_datetime(solar_df["timestamp"]).min().date()),
        end_date=str(pd.to_datetime(solar_df["timestamp"]).max().date()),
        frequency="5min",
    )
    try:
        solar_df.to_pickle(cache_path)
    except Exception:
        pass
    return solar_df


def fetch_nasa_power_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone_name: str,
    time_standard: str = "LST",
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    parameters = list(parameters or WEATHER_COLUMNS)
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    request_params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": pd.Timestamp(start_date).strftime("%Y%m%d"),
        "end": pd.Timestamp(end_date).strftime("%Y%m%d"),
        "time-standard": str(time_standard).upper(),
    }

    try:
        response = requests.get(base_url, params={**request_params, "format": "CSV", "header": "false"}, timeout=120)
        response.raise_for_status()
        weather = pd.read_csv(StringIO(response.text))
    except Exception:
        response = requests.get(base_url, params={**request_params, "format": "JSON"}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        parameter_block = payload["properties"]["parameter"]
        timestamp_keys = sorted(next(iter(parameter_block.values())).keys())
        base_timestamp = pd.to_datetime(timestamp_keys, format="%Y%m%d%H")
        weather = pd.DataFrame({"timestamp": base_timestamp})
        for parameter in parameters:
            weather[parameter] = [parameter_block[parameter].get(key, np.nan) for key in timestamp_keys]

    if "YEAR" in weather.columns:
        weather = weather.rename(columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"})
        weather["timestamp"] = pd.to_datetime(weather[["year", "month", "day", "hour"]])
        weather = weather.drop(columns=[col for col in ["year", "month", "day", "hour"] if col in weather.columns])

    weather = weather.loc[:, ["timestamp"] + [p for p in parameters if p in weather.columns]].copy()
    weather = weather.set_index("timestamp").sort_index()
    for column in parameters:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
        weather[column] = weather[column].replace(-999, np.nan)
    weather = weather.interpolate(method="time", limit_direction="both").ffill().bfill().fillna(0)
    weather.index.name = "timestamp"
    return weather.reset_index()


def align_weather_to_index(weather_hourly: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = weather_hourly.copy().set_index("timestamp").sort_index()
    aligned = aligned.reindex(aligned.index.union(target_index)).sort_index()
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].interpolate(method="time", limit_direction="both")
    aligned = aligned.reindex(target_index)
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].ffill().bfill()
    aligned.index.name = "timestamp"
    return aligned.reset_index()


def load_weather_hourly_cached(
    data_root: Path,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone_name: str,
    time_standard: str,
) -> pd.DataFrame:
    cache_dir = data_root / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weather_cache_path = cache_dir / WEATHER_CACHE_FILE
    weather_meta_path = cache_dir / WEATHER_CACHE_META_FILE

    request_start = pd.Timestamp(start_date)
    request_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    if weather_cache_path.exists() and weather_meta_path.exists():
        try:
            meta = json.loads(weather_meta_path.read_text(encoding="utf-8"))
            same_location = float(meta.get("latitude", 0.0)) == float(latitude) and float(meta.get("longitude", 0.0)) == float(longitude)
            same_standard = str(meta.get("time_standard", "")).upper() == str(time_standard).upper()
            if same_location and same_standard:
                cached_weather = pd.read_csv(weather_cache_path, parse_dates=["timestamp"])
                if not cached_weather.empty:
                    cached_start = pd.to_datetime(cached_weather["timestamp"]).min()
                    cached_end = pd.to_datetime(cached_weather["timestamp"]).max()
                    if cached_start <= request_start and cached_end >= request_end:
                        return cached_weather
        except Exception:
            pass

    weather = fetch_nasa_power_hourly(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone_name=timezone_name,
        time_standard=time_standard,
        parameters=WEATHER_COLUMNS,
    )

    try:
        weather.to_csv(weather_cache_path, index=False)
        weather_meta_path.write_text(
            json.dumps(
                {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "time_standard": str(time_standard).upper(),
                    "timezone": timezone_name,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    return weather


def load_latest_metrics(manifest_path: Path) -> dict:
    latest_metrics_path = manifest_path.parent.parent / "latest_metrics.json"
    if not latest_metrics_path.exists():
        return {}
    return json.loads(latest_metrics_path.read_text(encoding="utf-8"))


def select_model_name(manifest: dict, latest_metrics: dict, requested_model: str | None) -> str:
    saved_models = manifest.get("saved_models", {})
    if requested_model and requested_model in saved_models:
        return requested_model

    ranked_names = []
    for record in latest_metrics.get("one_step_metrics", []):
        model_name = record.get("Model") or record.get("model")
        if model_name:
            ranked_names.append(model_name)
    for model_name in ranked_names:
        if model_name in saved_models:
            return model_name

    if "Random Forest" in saved_models:
        return "Random Forest"
    return next(iter(saved_models.keys()))


def load_selected_model(manifest_path: Path, requested_model: str | None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest_metrics = load_latest_metrics(manifest_path)
    model_name = select_model_name(manifest, latest_metrics, requested_model)
    model_info = manifest["saved_models"][model_name]
    model_path = manifest_path.parent.parent / model_info["saved_path"] / "model.pkl"

    try:
        model = joblib.load(model_path)
    except Exception:
        fallback_name = "Random Forest"
        fallback_info = manifest["saved_models"].get(fallback_name)
        if fallback_info is None:
            raise
        model_path = manifest_path.parent.parent / fallback_info["saved_path"] / "model.pkl"
        model = joblib.load(model_path)
        model_name = fallback_name

    return model_name, model, manifest, latest_metrics


def compute_slot_stats(solar_rows):
    buckets = [[] for _ in range((24 * 60) // INTERVAL_MIN)]
    for row in solar_rows:
        buckets[slot_of(row["timestamp"])].append(row["solar_power_kw"])
    stats = []
    for arr in buckets:
        if not arr:
            stats.append((0.0, MIN_STD_KW))
            continue
        mean = float(sum(arr) / len(arr))
        if len(arr) > 1:
            var = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
            std = math.sqrt(var)
        else:
            std = MIN_STD_KW
        stats.append((mean, max(std, MIN_STD_KW)))
    return stats


def nearest_weather(weather_rows, ts: datetime):
    if not weather_rows:
        return {
            "timestamp": ts,
            "irradiance_wm2": 700.0,
            "cloud_pct": 35.0,
            "temp_c": 32.0,
            "source": "default",
        }
    best = min(weather_rows, key=lambda r: abs((r["timestamp"] - ts).total_seconds()))
    irr = best.get("irradiance_wm2", 700.0)
    cloud = best.get("cloud_pct", 35.0)
    temp = best.get("temp_c", 32.0)
    if not np.isfinite(irr):
        irr = 700.0
    if not np.isfinite(cloud):
        cloud = 35.0
    if not np.isfinite(temp):
        temp = 32.0
    return {
        "timestamp": best["timestamp"],
        "irradiance_wm2": float(irr),
        "cloud_pct": float(cloud),
        "temp_c": float(temp),
        "source": str(best.get("source", "nasa_power_nearest")),
    }


def nearest_solar_value(index, slot_means, ts: datetime, default=0.0):
    v = index.get(ts)
    if v is not None:
        return float(v)
    return float(slot_means[slot_of(ts)] if slot_means else default)


def rolling_from_index(index, slot_means, end_ts: datetime, steps: int):
    vals = []
    for i in range(1, steps + 1):
        ts = end_ts - timedelta(minutes=INTERVAL_MIN * i)
        vals.append(nearest_solar_value(index, slot_means, ts, 0.0))
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else MIN_STD_KW)


def rolling_weather_mean(weather_index, end_ts: datetime, steps: int):
    vals = []
    for i in range(1, steps + 1):
        ts = end_ts - timedelta(minutes=INTERVAL_MIN * i)
        w = weather_index.get(ts)
        if w is None:
            continue
        vals.append(float(w["irradiance_wm2"]))
    if not vals:
        return 700.0
    return float(np.mean(vals))


def build_feature_row(ts: datetime, weather, solar_index, slot_means, weather_index):
    irr = float(clamp(weather["irradiance_wm2"], 0.0, 1200.0))
    temp = float(weather["temp_c"])
    cloud = float(clamp(weather["cloud_pct"], 0.0, 100.0))

    hour = ts.hour + ts.minute / 60.0
    month = ts.month
    dow = ts.weekday()
    doy = ts.timetuple().tm_yday
    minute_of_day = ts.hour * 60 + ts.minute

    lag_1 = nearest_solar_value(solar_index, slot_means, ts - timedelta(minutes=5), 0.0)
    lag_24 = nearest_solar_value(solar_index, slot_means, ts - timedelta(minutes=120), 0.0)
    lag_288 = nearest_solar_value(solar_index, slot_means, ts - timedelta(minutes=1440), 0.0)

    roll_mean_3, _ = rolling_from_index(solar_index, slot_means, ts, 3)
    roll_mean_12, roll_std_12 = rolling_from_index(solar_index, slot_means, ts, 12)
    roll_mean_24, _ = rolling_from_index(solar_index, slot_means, ts, 24)
    irr_roll_12 = rolling_weather_mean(weather_index, ts, 12)

    return {
        "ALLSKY_SFC_SW_DWN": irr,
        "T2M": temp,
        "CLOUD_AMT": cloud,
        "hour": float(hour),
        "day": float(ts.day),
        "month": float(month),
        "day_of_week": float(dow),
        "day_of_year": float(doy),
        "minute_of_day": float(minute_of_day),
        "clock_hour": float(hour),
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "month_sin": math.sin(2 * math.pi * month / 12.0),
        "month_cos": math.cos(2 * math.pi * month / 12.0),
        "dow_sin": math.sin(2 * math.pi * dow / 7.0),
        "dow_cos": math.cos(2 * math.pi * dow / 7.0),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_288": lag_288,
        "rolling_mean_3": roll_mean_3,
        "rolling_mean_12": roll_mean_12,
        "rolling_mean_24": roll_mean_24,
        "rolling_std_12": max(roll_std_12, MIN_STD_KW),
        "irradiance_rolling_mean_12": irr_roll_12,
    }


def generate_prediction(payload: dict) -> dict:
    date_str = payload["date"]
    time_str = payload["time"]
    manifest_path = Path(payload["manifestPath"])
    data_root = Path(payload.get("dataRoot") or DEFAULT_DATA_ROOT)
    requested_model = payload.get("modelName") or payload.get("model")

    model_name, model, manifest, latest_metrics = load_selected_model(manifest_path, requested_model)
    feature_columns = manifest.get("config", {}).get("feature_columns", [])

    solar_df = load_solar_dataset_local(data_root)
    solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"]).dt.tz_localize(TZ)
    solar_df = solar_df.sort_values("timestamp").reset_index(drop=True)

    target_ts = parse_dt(f"{date_str}T{time_str}:00")
    min_ts = solar_df["timestamp"].min().to_pydatetime().astimezone(TZ)
    max_ts = solar_df["timestamp"].max().to_pydatetime().astimezone(TZ)

    # Keep the prediction timestamp as requested by the user. Only clamp the
    # history window end to available observed data for lag/rolling features.
    history_end_ts = target_ts
    if history_end_ts < min_ts:
        history_end_ts = min_ts
    if history_end_ts > max_ts:
        history_end_ts = max_ts

    history_cutoff = target_ts - timedelta(days=45)
    history_df = solar_df[(solar_df["timestamp"] >= history_cutoff) & (solar_df["timestamp"] <= history_end_ts)].copy()
    if history_df.empty:
        history_df = solar_df.copy()

    history = [
        {"timestamp": row.timestamp.to_pydatetime(), "solar_power_kw": float(row.power_kw)}
        for row in history_df.itertuples(index=False)
    ]
    slot_stats = compute_slot_stats(history)
    slot_means = [m for m, _ in slot_stats]
    solar_index = {r["timestamp"]: r["solar_power_kw"] for r in history}
    
    full_solar_index = {
        row.timestamp.to_pydatetime(): float(row.power_kw)
        for row in solar_df.itertuples(index=False)
    }

    weather_latitude = float(manifest.get("config", {}).get("latitude", 17.9374))
    weather_longitude = float(manifest.get("config", {}).get("longitude", 79.5960))
    weather_timezone = str(manifest.get("config", {}).get("timezone", "Asia/Kolkata"))
    weather_time_standard = str(manifest.get("config", {}).get("nasa_time_standard", "LST"))

    weather_rows_df = load_weather_hourly_cached(
        data_root=data_root,
        latitude=weather_latitude,
        longitude=weather_longitude,
        start_date=str(history_df["timestamp"].min().date()),
        end_date=str(target_ts.date()),
        timezone_name=weather_timezone,
        time_standard=weather_time_standard,
    )
    weather_rows_df["timestamp"] = pd.to_datetime(weather_rows_df["timestamp"])
    if weather_rows_df["timestamp"].dt.tz is None:
        weather_rows_df["timestamp"] = weather_rows_df["timestamp"].dt.tz_localize(TZ)
    else:
        weather_rows_df["timestamp"] = weather_rows_df["timestamp"].dt.tz_convert(TZ)
    weather_start = history_df["timestamp"].min()
    weather_end = target_ts.replace(hour=23, minute=59, second=59, microsecond=0)
    weather_rows_df = weather_rows_df[
        (weather_rows_df["timestamp"] >= weather_start) & (weather_rows_df["timestamp"] <= weather_end)
    ].copy()
    weather_rows = [
        {
            "timestamp": row.timestamp.to_pydatetime(),
            "irradiance_wm2": float(getattr(row, "ALLSKY_SFC_SW_DWN")),
            "cloud_pct": float(getattr(row, "CLOUD_AMT")),
            "temp_c": float(getattr(row, "T2M")),
            "source": "nasa_power",
        }
        for row in weather_rows_df.itertuples(index=False)
    ]

    target_date = target_ts.date()
    target_day_rows = [w for w in weather_rows if w["timestamp"].date() == target_date]
    target_day_max_irr = max((float(w["irradiance_wm2"]) for w in target_day_rows), default=0.0)
    target_day_daylight_points = sum(1 for w in target_day_rows if float(w["irradiance_wm2"]) > DAY_IRRADIANCE_THRESHOLD)
    weather_mode = "nasa_power"

    # If target day weather is missing/sparse/unusable (common for near-real-time future dates),
    # synthesize target-day weather from the most recent valid daylight profile.
    if (
        not target_day_rows
        or target_day_max_irr <= DAY_IRRADIANCE_THRESHOLD
        or target_day_daylight_points < 6
    ):
        rows_by_date = {}
        for row in weather_rows:
            rows_by_date.setdefault(row["timestamp"].date(), []).append(row)

        candidate_dates = []
        for d, rows in rows_by_date.items():
            max_irr = max((float(r["irradiance_wm2"]) for r in rows), default=0.0)
            if max_irr > DAY_IRRADIANCE_THRESHOLD:
                candidate_dates.append(d)

        if candidate_dates:
            proxy_date = max(candidate_dates)
            proxy_rows = []
            for row in rows_by_date[proxy_date]:
                src_ts = row["timestamp"]
                shifted_ts = datetime(
                    year=target_ts.year,
                    month=target_ts.month,
                    day=target_ts.day,
                    hour=src_ts.hour,
                    minute=src_ts.minute,
                    second=0,
                    microsecond=0,
                    tzinfo=TZ,
                )
                proxy_rows.append({
                    "timestamp": shifted_ts,
                    "irradiance_wm2": float(row["irradiance_wm2"]),
                    "cloud_pct": float(row["cloud_pct"]),
                    "temp_c": float(row["temp_c"]),
                    "source": "recent_profile_proxy",
                })

            weather_rows = [w for w in weather_rows if w["timestamp"].date() != target_date]
            weather_rows.extend(proxy_rows)
            weather_mode = "recent_profile_proxy"
        else:
            # Final fallback when NASA day-level irradiance is flat/missing:
            # derive a synthetic daylight profile from learned slot means.
            synthetic_profile = np.array(slot_means, dtype=float)
            if float(np.nanmax(synthetic_profile)) <= 0.0:
                full_history = [
                    {"timestamp": row.timestamp.to_pydatetime(), "solar_power_kw": float(row.power_kw)}
                    for row in solar_df.itertuples(index=False)
                ]
                full_slot_stats = compute_slot_stats(full_history)
                synthetic_profile = np.array([m for m, _ in full_slot_stats], dtype=float)

            profile_peak = float(np.nanmax(synthetic_profile)) if synthetic_profile.size else 0.0
            if profile_peak <= 0.0:
                profile_peak = 1.0

            proxy_rows = []
            for hour in range(24):
                ts = datetime(
                    year=target_ts.year,
                    month=target_ts.month,
                    day=target_ts.day,
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=TZ,
                )
                slot_idx = (hour * 60) // INTERVAL_MIN
                rel = float(max(0.0, synthetic_profile[slot_idx] / profile_peak))
                irr = rel * 950.0
                proxy_rows.append({
                    "timestamp": ts,
                    "irradiance_wm2": float(irr),
                    "cloud_pct": float(35.0 if irr > DAY_IRRADIANCE_THRESHOLD else 70.0),
                    "temp_c": float(32.0 if irr > DAY_IRRADIANCE_THRESHOLD else 27.0),
                    "source": "synthetic_profile_proxy",
                })

            weather_rows = [w for w in weather_rows if w["timestamp"].date() != target_date]
            weather_rows.extend(proxy_rows)
            weather_mode = "synthetic_profile_proxy"

    weather_index = {w["timestamp"]: w for w in weather_rows}

    day_start = target_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    steps = (24 * 60) // INTERVAL_MIN

    labels = []
    solar_pred = []
    observed_power = []
    low_band = []
    high_band = []
    meta = []
    selected_idx = 0
    points = []
    prediction_mode = "model"

    for i in range(steps):
        ts = day_start + timedelta(minutes=i * INTERVAL_MIN)
        weather = nearest_weather(weather_rows, ts)
        feat = build_feature_row(ts, weather, solar_index, slot_means, weather_index)
        X = np.array([[float(feat.get(c, 0.0)) for c in feature_columns]], dtype=float)
        y = float(model.predict(X)[0]) * CALIBRATION_FACTOR
        if feat["ALLSKY_SFC_SW_DWN"] <= DAY_IRRADIANCE_THRESHOLD:
            y = 0.0
        else:
            y = max(0.0, y)

        slot_mean, slot_std = slot_stats[slot_of(ts)]
        sigma = max(float(slot_std) * CALIBRATION_FACTOR, MIN_STD_KW)
        lo = max(0.0, y - 1.28 * sigma)
        hi = y + 1.28 * sigma

        labels.append(fmt_time(ts))
        solar_pred.append(round2(y))
        
        obs_val = full_solar_index.get(ts)
        observed_power.append(round2(obs_val) if obs_val is not None else None)
        
        low_band.append(round2(lo))
        high_band.append(round2(hi))
        meta.append({
            "timestamp": to_iso(ts),
            "irradiance_wm2": round2(weather["irradiance_wm2"]),
            "cloud_pct": round2(weather["cloud_pct"]),
            "temp_c": round2(weather["temp_c"]),
            "status": "active" if y > 1 else "standby",
            "slot_mean_kw": round2(slot_mean),
            "slot_std_kw": round2(slot_std),
        })

        if abs((ts - target_ts).total_seconds()) <= INTERVAL_MIN * 60:
            selected_idx = i

    daylight_available = any(float(m.get("irradiance_wm2", 0.0)) > DAY_IRRADIANCE_THRESHOLD for m in meta)
    if daylight_available and max(solar_pred or [0.0]) <= 0.0:
        prediction_mode = "slot_weather_proxy"
        solar_pred = []
        observed_power = []
        low_band = []
        high_band = []
        meta = []

        for i in range(steps):
            ts = day_start + timedelta(minutes=i * INTERVAL_MIN)
            weather = nearest_weather(weather_rows, ts)
            slot_mean, slot_std = slot_stats[slot_of(ts)]

            irr = float(clamp(weather["irradiance_wm2"], 0.0, 1200.0))
            cloud = float(clamp(weather["cloud_pct"], 0.0, 100.0))
            temp = float(weather["temp_c"])

            irr_factor = 0.55 + 0.75 * (irr / 1000.0)
            cloud_factor = 1.0 - 0.45 * (cloud / 100.0)
            temp_factor = 1.0 - min(0.18, abs(temp - 32.0) * 0.005)

            y = 0.0 if irr <= DAY_IRRADIANCE_THRESHOLD else max(0.0, float(slot_mean) * irr_factor * cloud_factor * temp_factor * CALIBRATION_FACTOR)
            sigma = max(float(slot_std) * CALIBRATION_FACTOR, MIN_STD_KW)
            lo = max(0.0, y - 1.28 * sigma)
            hi = y + 1.28 * sigma

            solar_pred.append(round2(y))
            
            obs_val = full_solar_index.get(ts)
            observed_power.append(round2(obs_val) if obs_val is not None else None)
            
            low_band.append(round2(lo))
            high_band.append(round2(hi))
            meta.append({
                "timestamp": to_iso(ts),
                "irradiance_wm2": round2(weather["irradiance_wm2"]),
                "cloud_pct": round2(weather["cloud_pct"]),
                "temp_c": round2(weather["temp_c"]),
                "status": "active" if y > 1 else "standby",
                "slot_mean_kw": round2(slot_mean) if 'slot_mean' in locals() else 0.0,
                "slot_std_kw": round2(slot_std) if 'slot_std' in locals() else 0.0,
            })

    current = solar_pred[selected_idx]
    current_obs = observed_power[selected_idx]
    current_low = low_band[selected_idx]
    current_high = high_band[selected_idx]

    target_weather = nearest_weather(weather_rows, target_ts)
    points.append({
        "timestamp": to_iso(target_ts),
        "timeLabel": fmt_time(target_ts),
        "solar_power_kw": round2(current),
        "observed_power_kw": current_obs,
        "confidence_low_kw": round2(current_low),
        "confidence_high_kw": round2(current_high),
        "irradiance_proxy": round2(target_weather["irradiance_wm2"]),
        "cloud_pct": round2(target_weather["cloud_pct"]),
        "temp_c": round2(target_weather["temp_c"]),
        "status": "active" if current > 1 else "standby",
        "source": model_name,
    })

    return {
        "ok": True,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "requestDate": fmt_date(target_ts),
        "requestTime": fmt_time(target_ts),
        "resolvedModel": model_name,
        "dataSource": f"local_xlsx + {weather_mode}",
        "note": (
            f"Prediction is generated by the trained {model_name} model from the saved outputs."
            if prediction_mode == "model"
            else f"Prediction used {model_name} model context with dynamic slot-weather fallback for this timestamp."
        ),
        "summary": {
            "currentSolarKw": round2(current),
            "currentObservedKw": current_obs,
            "peakSolarKw": round2(max(solar_pred) if solar_pred else current),
            "avgSolarKw": round2(float(np.mean(solar_pred)) if solar_pred else current),
            "confidenceBandKw": {"low": round2(current_low), "high": round2(current_high)},
            "status": "active" if current > 1 else "standby",
        },
        "chartSeries": {
            "labels": labels,
            "solarPowerKw": solar_pred,
            "observedPowerKw": observed_power,
            "lowBandKw": low_band,
            "highBandKw": high_band,
            "selectedIdx": int(selected_idx),
            "meta": meta,
        },
        "forecastColumns": points,
        "selectedMetrics": latest_metrics.get("one_step_metrics", []),
    }


def main():
    payload = json.loads(sys.stdin.read())
    result = generate_prediction(payload)
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
