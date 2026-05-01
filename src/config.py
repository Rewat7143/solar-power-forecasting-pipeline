"""
Pipeline configuration, constants, and global setup.

Centralises all hyperparameters, data paths, and environment flags
so that every other module can ``from src.config import CONFIG``.
"""

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Optional dependency ──────────────────────────────────────────────
try:
    from lightgbm import LGBMRegressor  # noqa: F401

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ── Pipeline configuration ───────────────────────────────────────────
CONFIG = {
    # Data source
    "data_source_mode": "local_xlsx",  # "local_xlsx" or "google_sheets"
    "workspace_root": Path("."),
    "local_xlsx_glob": "data/**/Inst_*.xlsx",
    "spreadsheet_id_or_url": None,
    "google_sheet_tabs": [
        "MTR_24", "MTR_25", "MTR_26", "MTR_27", "MTR_28", "MTR_29", "Gen_Sum",
    ],
    "google_csv_urls": {},
    "meter_tabs": ["MTR_24", "MTR_25", "MTR_26", "MTR_27", "MTR_28", "MTR_29"],
    "summary_tab": "Gen_Sum",

    # Date range and location
    "date_range": ("2025-07-14", "2026-04-12"),
    "latitude": 17.9374,
    "longitude": 79.5960,
    "timezone": "Asia/Kolkata",
    "nasa_time_standard": "LST",

    # Signal processing
    "target_column": "power_kw",
    "frequency": "5min",
    "night_power_threshold_kw": 0.05,
    "day_irradiance_threshold": 20.0,
    "daylight_hour_window": (5.0, 19.5),

    # Sequence modelling
    "sequence_length": 48,  # 48 × 5 min = 4 hours
    "forecast_horizon_hours": 24,
    "backtest_horizon_hours": 24,
    "backtest_stride_hours": 24,
    "random_seed": 42,

    # Model names
    "model_names": [
        "Random Forest",
        "LightGBM",
        "CNN + LSTM",
        "PatchTST",
        "Temporal Fusion Transformer",
        "CNN + Transformer",
    ],

    # Tree model hyperparameter grids
    "rf_param_grid": [
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 18, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.7},
    ],
    "lgbm_param_grid": [
        {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.8},
    ],

    # Deep learning training
    "dl_batch_size": 256,
    "dl_learning_rate": 1e-3,
    "dl_weight_decay": 1e-5,
    "dl_max_epochs": 24,
    "dl_patience": 5,

    # Transformer / PatchTST architecture
    "patch_len": 6,
    "patch_stride": 3,
    "transformer_heads": 4,
    "transformer_layers": 2,
    "transformer_d_model": 64,

    # Ensemble
    "stacking_top_k": 4,
    "stacking_meta_train_fraction": 0.70,

    # Train / val / test split
    "train_fraction": 0.70,
    "val_fraction": 0.15,
}


# ── Derived constants ────────────────────────────────────────────────
WEATHER_COLUMNS = ["ALLSKY_SFC_SW_DWN", "T2M", "CLOUD_AMT"]
STEPS_PER_HOUR = int(pd.Timedelta("1h") / pd.Timedelta(CONFIG["frequency"]))
FORECAST_STEPS = CONFIG["forecast_horizon_hours"] * STEPS_PER_HOUR


# ── Reproducibility ──────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LightGBM is disabled on Python ≥ 3.14 due to kernel stability issues.
LIGHTGBM_ENABLED = HAS_LIGHTGBM and sys.version_info < (3, 14)

# Suppress noisy warnings during notebook / script runs.
warnings.filterwarnings("ignore")
