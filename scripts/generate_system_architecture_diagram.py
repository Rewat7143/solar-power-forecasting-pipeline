from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT_DIR = Path("output_images_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 8), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(
    50,
    95,
    "Solar Power Forecasting System Architecture",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
    color="#111827",
)
ax.text(
    50,
    90,
    "End-to-end data flow from ingestion to prediction output",
    ha="center",
    va="center",
    fontsize=12,
    color="#4b5563",
)

# Node definitions: x, y, w, h, color, title, subtitle
nodes = [
    (4, 58, 18, 18, "#dbeafe", "Solar Meter Data", "Local Excel / CSV\nMTR_24 to MTR_29\nGen_Sum total kW"),
    (4, 28, 18, 18, "#e0e7ff", "NASA Weather Data", "ALLSKY_SFC_SW_DWN\nT2M\nCLOUD_AMT"),
    (28, 43, 20, 20, "#dcfce7", "Preprocessing", "Clean timestamps\nRemove summaries\nAlign frequency\nHandle missing values"),
    (52, 43, 20, 20, "#fef3c7", "Feature Engineering", "Lag features\nRolling statistics\nCyclical time encoding\nWeather + solar features"),
    (76, 43, 20, 20, "#fce7f3", "Model Layer", "Random Forest\nPatchTST\nCNN + LSTM\nCNN + Transformer\nTFT + Ensemble"),
    (76, 12, 20, 16, "#ecfccb", "Prediction Output", "Forecast power (kW)\n24-hour forecast\nGradio UI / saved files"),
]

# Draw nodes
for x, y, w, h, color, title, subtitle in nodes:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=8",
        linewidth=1.6,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12, fontweight="bold", color="#111827")
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=9.5, color="#1f2937", linespacing=1.35)

# Helper for arrows between boxes

def arrow(x1, y1, x2, y2, color="#475569"):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=2,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)

# Data flow arrows
arrow(22, 67, 28, 58)   # solar meter -> preprocessing
arrow(22, 37, 28, 48)   # weather -> preprocessing
arrow(48, 53, 52, 53)   # preprocessing -> feature engineering
arrow(72, 53, 76, 53)   # feature engineering -> models
arrow(86, 43, 86, 28)   # models -> output

# Merge label
ax.text(26, 78, "Data Ingestion", fontsize=11, fontweight="bold", color="#0f172a")
ax.text(26, 21, "External Weather API", fontsize=11, fontweight="bold", color="#0f172a")

# Bottom explanatory band
band = FancyBboxPatch((4, 4), 92, 6, boxstyle="round,pad=0.02,rounding_size=4", linewidth=1.2, edgecolor="#cbd5e1", facecolor="#f8fafc")
ax.add_patch(band)
ax.text(
    50,
    7,
    "Pipeline purpose: combine physical solar measurements with weather drivers to produce short-term and 24-hour solar power forecasts",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#334155",
)

output_path = OUTPUT_DIR / "system_architecture_diagram.png"
fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {output_path.resolve()}")
