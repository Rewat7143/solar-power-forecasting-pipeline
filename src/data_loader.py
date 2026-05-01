"""
Data ingestion from local Excel workbooks and Google Sheets.

Handles reading solar meter data from multiple tabs (MTR_24–MTR_29, Gen_Sum),
cleaning timestamps, merging meter readings, and producing a unified
time-series DataFrame at 5-minute resolution.
"""

from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Sequence
from urllib.parse import quote

import pandas as pd
import requests


# ── Google Sheets helpers ────────────────────────────────────────────

def extract_spreadsheet_id(spreadsheet_id_or_url: Optional[str]) -> Optional[str]:
    """Extract a Google Sheets spreadsheet ID from a URL or pass-through string."""
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
    """Build a direct CSV export URL for a single Google Sheet tab."""
    safe_sheet = quote(sheet_name, safe="")
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={safe_sheet}"
    )


def fetch_public_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    """Download a publicly accessible CSV from *url* and return a DataFrame."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


# ── Sheet cleaning ───────────────────────────────────────────────────

def clean_timeseries_sheet(
    frame: pd.DataFrame,
    power_column: str,
    rename_to: str = "power_kw",
) -> pd.DataFrame:
    """Standardise a raw meter sheet to ``[timestamp, power_kw]``."""
    if "Timestamp" not in frame.columns or power_column not in frame.columns:
        return pd.DataFrame(columns=["timestamp", rename_to])

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["Timestamp"], errors="coerce")
    cleaned[rename_to] = pd.to_numeric(cleaned[power_column], errors="coerce")
    cleaned = cleaned.loc[cleaned["timestamp"].notna(), ["timestamp", rename_to]].copy()
    cleaned = cleaned.dropna(subset=[rename_to])
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return cleaned.reset_index(drop=True)


# ── Google Sheets loader ─────────────────────────────────────────────

def load_google_solar_data(
    spreadsheet_id_or_url: Optional[str],
    meter_tabs: Sequence[str],
    summary_tab: str,
    csv_url_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load solar generation data from Google Sheets (summary tab preferred)."""
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

    # Try summary tab first
    try:
        summary_df = get_tab_df(summary_tab)
        summary_clean = clean_timeseries_sheet(
            summary_df, power_column="Solar Total kW", rename_to="power_kw",
        )
        if not summary_clean.empty:
            summary_clean["source"] = "google_sheet_summary"
            return summary_clean
    except Exception as exc:
        print(f"Summary tab fallback triggered: {exc}")

    # Fall back to individual meter tabs
    meter_frames = []
    for tab_name in meter_tabs:
        meter_df = get_tab_df(tab_name)
        meter_clean = clean_timeseries_sheet(
            meter_df, power_column="kW_Total", rename_to=f"{tab_name}_power_kw",
        )
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


# ── Local Excel loader ───────────────────────────────────────────────

def load_solar_from_single_workbook(
    workbook_path: Path,
    meter_tabs: Sequence[str],
    summary_tab: str,
) -> pd.DataFrame:
    """Parse a single Excel workbook and return a unified power DataFrame."""
    excel_file = pd.ExcelFile(workbook_path)

    if summary_tab in excel_file.sheet_names:
        summary_df = excel_file.parse(summary_tab)
        summary_clean = clean_timeseries_sheet(
            summary_df, power_column="Solar Total kW", rename_to="power_kw",
        )
        if not summary_clean.empty:
            summary_clean["source_file"] = workbook_path.name
            summary_clean["source"] = "local_summary"
            return summary_clean

    meter_frames = []
    for tab_name in meter_tabs:
        if tab_name not in excel_file.sheet_names:
            continue
        meter_df = excel_file.parse(tab_name)
        meter_clean = clean_timeseries_sheet(
            meter_df, power_column="kW_Total", rename_to=f"{tab_name}_power_kw",
        )
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


# ── Finalise and load ────────────────────────────────────────────────

def finalize_solar_dataframe(
    solar_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str,
) -> pd.DataFrame:
    """Clip to date range, reindex to regular frequency, interpolate gaps."""
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
    full_index = pd.date_range(
        start=solar_df["timestamp"].min(),
        end=solar_df["timestamp"].max(),
        freq=frequency,
    )
    solar_df = solar_df.set_index("timestamp").reindex(full_index)
    solar_df.index.name = "timestamp"
    solar_df["missing_power_flag"] = solar_df["power_kw"].isna().astype(int)
    solar_df["power_kw"] = solar_df["power_kw"].interpolate(
        method="time", limit=2, limit_direction="both",
    )
    solar_df["power_kw"] = solar_df["power_kw"].clip(lower=0)
    metadata_columns = [
        col for col in solar_df.columns if col not in {"power_kw", "missing_power_flag"}
    ]
    for column in metadata_columns:
        solar_df[column] = solar_df[column].ffill().bfill()
    solar_df = solar_df.dropna(subset=["power_kw"]).reset_index()
    return solar_df


def load_solar_dataset(config: dict) -> pd.DataFrame:
    """Top-level loader that dispatches to Google Sheets or local Excel."""
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
            raise FileNotFoundError(
                "No local Excel workbooks matched the configured glob pattern."
            )

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
