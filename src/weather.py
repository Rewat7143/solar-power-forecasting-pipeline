"""
NASA POWER weather data acquisition and alignment.

Fetches hourly weather variables (irradiance, temperature, cloud cover) from
the NASA POWER API, handles CSV/JSON fallback, and interpolates to match the
5-minute resolution of the solar meter data.
"""

from io import StringIO
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests

from src.config import WEATHER_COLUMNS


def fetch_nasa_power_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    time_standard: str = "LST",
    parameters: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Fetch hourly weather from the NASA POWER API (CSV with JSON fallback)."""
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
        weather = pd.DataFrame({"timestamp": base_timestamp})
        for parameter in parameters:
            weather[parameter] = [
                parameter_block[parameter].get(key, np.nan) for key in timestamp_keys
            ]

    # Clean and interpolate
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
    """Upsample hourly weather to the target 5-minute index via interpolation."""
    aligned = weather_hourly.copy().set_index("timestamp").sort_index()
    aligned = aligned.reindex(aligned.index.union(target_index)).sort_index()
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].interpolate(
        method="time", limit_direction="both",
    )
    aligned = aligned.reindex(target_index)
    aligned[WEATHER_COLUMNS] = aligned[WEATHER_COLUMNS].ffill().bfill()
    aligned.index.name = "timestamp"
    return aligned.reset_index()


def merge_solar_and_weather(
    solar_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join solar meter data with aligned weather features."""
    merged = solar_df.merge(weather_df, on="timestamp", how="left").sort_values("timestamp")
    for column in WEATHER_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = merged[column].interpolate(method="linear", limit_direction="both")
        merged[column] = merged[column].ffill().bfill()

    merged["power_kw"] = pd.to_numeric(merged["power_kw"], errors="coerce").clip(lower=0)
    merged = merged.dropna(subset=["timestamp", "power_kw"] + WEATHER_COLUMNS)
    return merged.reset_index(drop=True)
