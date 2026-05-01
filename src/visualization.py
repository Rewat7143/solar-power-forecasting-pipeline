"""
Visualization helpers for model evaluation and reporting.

All plot functions use matplotlib and follow a consistent style with
the project's colour palette.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Set a clean, consistent plot style
plt.style.use("seaborn-v0_8-whitegrid")

# Project colour palette
PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0f766e"]
ACTUAL_COLOR = "#1f2937"
FORECAST_COLOR = "#059669"
FORECAST_FILL = "#10b981"
BACKTEST_COLOR = "#0f766e"


def plot_model_comparison(
    test_df: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    title: str,
    top_k: int = 4,
) -> None:
    """Plot actual vs. predicted for multiple models on the test set."""
    top_columns = [col for col in prediction_frame.columns if col != "timestamp"][:top_k]
    plt.figure(figsize=(16, 6))
    plt.plot(
        test_df["timestamp"], test_df["power_kw"],
        label="Actual", linewidth=2, color=ACTUAL_COLOR,
    )
    for idx, column in enumerate(top_columns):
        plt.plot(
            prediction_frame["timestamp"], prediction_frame[column],
            label=column, alpha=0.85, color=PALETTE[idx % len(PALETTE)],
        )
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_recursive_backtest(
    backtest_df: pd.DataFrame,
    model_label: str,
) -> None:
    """Plot actual vs. recursive forecast for daylight hours."""
    plot_df = backtest_df.loc[backtest_df["is_daylight"] == 1].sort_values("timestamp").copy()
    plt.figure(figsize=(16, 6))
    plt.plot(
        plot_df["timestamp"], plot_df["actual_power_kw"],
        label="Actual", linewidth=2, color=ACTUAL_COLOR,
    )
    plt.plot(
        plot_df["timestamp"], plot_df["forecast_power_kw"],
        label=f"{model_label} recursive backtest",
        linewidth=1.6, color=BACKTEST_COLOR,
    )
    plt.title(f"Rolling 24-Hour Recursive Backtest: {model_label}")
    plt.xlabel("Timestamp")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_future_forecast(forecast_df: pd.DataFrame) -> None:
    """Plot the next-24h solar power forecast with confidence fill."""
    plt.figure(figsize=(16, 5))
    plt.plot(
        forecast_df["timestamp"], forecast_df["forecast_power_kw"],
        linewidth=2.5, color=FORECAST_COLOR, label="Forecasted solar power",
    )
    plt.fill_between(
        forecast_df["timestamp"], forecast_df["forecast_power_kw"],
        alpha=0.20, color=FORECAST_FILL,
    )
    plt.title("Next 24 Hours Solar Power Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Forecast Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()
