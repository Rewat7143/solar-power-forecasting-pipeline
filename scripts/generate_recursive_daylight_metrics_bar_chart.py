from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("output_images_v3")
OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_FILE = Path("latest_metrics.json")

if not METRICS_FILE.exists():
    raise FileNotFoundError("latest_metrics.json not found. Run the pipeline first.")

payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
rows = payload.get("recursive_daylight", [])
if not rows:
    raise ValueError("No recursive_daylight section found in latest_metrics.json")

frame = pd.DataFrame(rows)
required = {"Model", "MAE", "RMSE", "R2"}
missing = required.difference(frame.columns)
if missing:
    raise ValueError(f"Missing required columns in recursive_daylight: {sorted(missing)}")

frame = frame[["Model", "MAE", "RMSE", "R2"]].copy()
frame = frame.sort_values("RMSE").reset_index(drop=True)

x = np.arange(len(frame))
width = 0.25

fig, ax1 = plt.subplots(figsize=(19, 8), dpi=200)
fig.patch.set_facecolor("white")

mae_bars = ax1.bar(x - width, frame["MAE"], width, label="MAE (kW)", color="#2563eb")
rmse_bars = ax1.bar(x, frame["RMSE"], width, label="RMSE (kW)", color="#dc2626")
ax1.set_ylabel("Error (kW)")
ax1.set_xlabel("Model")
ax1.set_xticks(x)
ax1.set_xticklabels(frame["Model"], rotation=18, ha="right")
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("Recursive Daylight Backtest: MAE, RMSE, and R²")

ax2 = ax1.twinx()
r2_bars = ax2.bar(x + width, frame["R2"], width, label="R² Score", color="#059669")
ax2.set_ylabel("R² Score")
ax2.axhline(0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.8)


def annotate_bars(ax, bars, fmt, offset=0.2):
    for bar in bars:
        height = float(bar.get_height())
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )


annotate_bars(ax1, mae_bars, "{:.1f}", offset=0.8)
annotate_bars(ax1, rmse_bars, "{:.1f}", offset=0.8)
for bar in r2_bars:
    height = float(bar.get_height())
    delta = 0.03 if height >= 0 else -0.05
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + delta,
        f"{height:.3f}",
        ha="center",
        va="bottom" if height >= 0 else "top",
        fontsize=8,
    )

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

fig.text(
    0.5,
    0.02,
    "Metrics from recursive 24-hour daylight backtest. Lower MAE/RMSE is better; higher R² is better.",
    ha="center",
    fontsize=10,
    color="#555555",
)

fig.tight_layout(rect=(0, 0.04, 1, 0.97))
output_path = OUTPUT_DIR / "recursive_daylight_metrics_bar_chart.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path.resolve()}")
