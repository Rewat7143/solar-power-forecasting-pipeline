from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("output_images_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

# Metrics aligned with the comparison table image.
# Keep this list in the same rank order as the table.
data = [
    {"Model": "Random Forest", "MAE": 7.84, "RMSE": 15.96, "R2": 0.9456},
    {"Model": "Ensemble (RF + PatchTST)", "MAE": 10.12, "RMSE": 20.34, "R2": 0.9234},
    {"Model": "PatchTST", "MAE": 12.48, "RMSE": 24.72, "R2": 0.8923},
    {"Model": "CNN + LSTM", "MAE": 14.12, "RMSE": 28.34, "R2": 0.8612},
    {"Model": "CNN + Transformer", "MAE": 15.89, "RMSE": 31.07, "R2": 0.8401},
    {"Model": "Temporal Fusion Transformer", "MAE": 16.54, "RMSE": 33.21, "R2": 0.8145},
]

frame = pd.DataFrame(data)
frame["Accuracy %"] = frame["R2"] * 100.0

models = frame["Model"].tolist()
x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(18, 8), dpi=200)
fig.patch.set_facecolor("white")

mae_bars = ax1.bar(x - width, frame["MAE"], width, label="MAE (kW)", color="#2563eb")
rmse_bars = ax1.bar(x, frame["RMSE"], width, label="RMSE (kW)", color="#dc2626")
ax1.set_ylabel("Error (kW)")
ax1.set_xlabel("Model")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=18, ha="right")
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("Model Comparison Bar Chart: MAE, RMSE, and R² Score")

ax2 = ax1.twinx()
r2_bars = ax2.bar(x + width, frame["R2"], width, label="R² Score", color="#059669")
ax2.set_ylabel("R² Score")
ax2.axhline(0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.7)

# Annotate bars for readability.
def annotate_bars(ax, bars, fmt, offset=0.2):
    for bar in bars:
        height = bar.get_height()
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
            rotation=0,
        )

annotate_bars(ax1, mae_bars, "{:.1f}", offset=0.8)
annotate_bars(ax1, rmse_bars, "{:.1f}", offset=0.8)
for bar in r2_bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.03 if height >= 0 else -0.05),
        f"{height:.3f}",
        ha="center",
        va="bottom" if height >= 0 else "top",
        fontsize=8,
    )

# Combined legend.
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

fig.text(
    0.5,
    0.02,
    "Metrics are aligned with the comparison table. Lower MAE/RMSE is better; higher R² is better.",
    ha="center",
    fontsize=10,
    color="#555555",
)

fig.tight_layout(rect=(0, 0.04, 1, 0.97))
output_path = OUTPUT_DIR / "model_metrics_bar_chart.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path.resolve()}")
