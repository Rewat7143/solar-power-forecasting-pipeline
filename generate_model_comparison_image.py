from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("output_images_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

# Latest reported evaluation values from the notebook/report.
# For this regression project, "accuracy" is shown as R²-based accuracy (R² × 100).
data = [
    {
        "Model": "Random Forest",
        "Family": "Tree Ensemble",
        "RMSE (kW)": 15.96,
        "MAE (kW)": 7.84,
        "R²": 0.9456,
        "Accuracy %": 94.56,
    },
    {
        "Model": "Ensemble (RF + PatchTST)",
        "Family": "Weighted Blend",
        "RMSE (kW)": 20.34,
        "MAE (kW)": 10.12,
        "R²": 0.9234,
        "Accuracy %": 92.34,
    },
    {
        "Model": "PatchTST",
        "Family": "Transformer",
        "RMSE (kW)": 24.72,
        "MAE (kW)": 12.48,
        "R²": 0.8923,
        "Accuracy %": 89.23,
    },
    {
        "Model": "CNN + LSTM",
        "Family": "Hybrid",
        "RMSE (kW)": 28.34,
        "MAE (kW)": 14.12,
        "R²": 0.8612,
        "Accuracy %": 86.12,
    },
    {
        "Model": "CNN + Transformer",
        "Family": "Hybrid",
        "RMSE (kW)": 31.07,
        "MAE (kW)": 15.89,
        "R²": 0.8401,
        "Accuracy %": 84.01,
    },
    {
        "Model": "Temporal Fusion Transformer",
        "Family": "Transformer",
        "RMSE (kW)": 33.21,
        "MAE (kW)": 16.54,
        "R²": 0.8145,
        "Accuracy %": 81.45,
    },
]

frame = pd.DataFrame(data)
frame = frame.sort_values("RMSE (kW)").reset_index(drop=True)
frame["Rank"] = frame.index + 1
frame = frame[["Rank", "Model", "Family", "RMSE (kW)", "MAE (kW)", "R²", "Accuracy %"]]

# Build a clean comparison table image.
fig, ax = plt.subplots(figsize=(18, 6.5), dpi=200)
fig.patch.set_facecolor("white")
ax.axis("off")

# Title and subtitle.
fig.suptitle(
    "Solar Power Forecasting Model Comparison",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)
ax.set_title(
    "Latest execution results | Accuracy shown as R²-based accuracy (R² × 100)",
    fontsize=11,
    color="#444444",
    pad=18,
)

cell_text = []
for _, row in frame.iterrows():
    cell_text.append([
        int(row["Rank"]),
        row["Model"],
        row["Family"],
        f'{row["RMSE (kW)"]:.2f}',
        f'{row["MAE (kW)"]:.2f}',
        f'{row["R²"]:.4f}',
        f'{row["Accuracy %"]:.2f}%',
    ])

col_labels = ["Rank", "Model", "Family", "RMSE (kW)", "MAE (kW)", "R²", "Accuracy %"]
col_widths = [0.05, 0.25, 0.16, 0.12, 0.12, 0.10, 0.10]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    colWidths=col_widths,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Header styling.
header_color = "#2f6fed"
for c in range(len(col_labels)):
    cell = table[(0, c)]
    cell.set_facecolor(header_color)
    cell.set_text_props(color="white", weight="bold")
    cell.set_edgecolor("white")

# Body styling with highlight for best model.
for r in range(1, len(frame) + 1):
    for c in range(len(col_labels)):
        cell = table[(r, c)]
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.8)
        cell.set_facecolor("#f7fbff" if r % 2 else "#eef5ff")
        if r == 1:
            cell.set_facecolor("#dff2df")
            if c == 1:
                cell.set_text_props(weight="bold")

# Footer note.
fig.text(
    0.5,
    0.03,
    "Regression note: accuracy is not a native metric for regression, so R² is converted to a percentage for presentation.",
    ha="center",
    fontsize=10,
    color="#555555",
)

output_path = OUTPUT_DIR / "model_comparison_table.png"
plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Saved: {output_path.resolve()}")
