from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("output_images_v3")
OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_FILE = Path("latest_metrics.json")

# Latest reported evaluation values from the notebook/report.
# For this regression project, "accuracy" is shown as R²-based accuracy (R² × 100).
data = [
    {
        "Model": "Random Forest",
        "Family": "Tree Ensemble",
        "RMSE (kW)": 6.9801,
        "MAE (kW)": 3.3359,
        "R²": 0.9753,
        "Accuracy %": 97.53,
    },
    {
        "Model": "PatchTST",
        "Family": "Transformer",
        "RMSE (kW)": 11.9409,
        "MAE (kW)": 7.4884,
        "R²": 0.9276,
        "Accuracy %": 92.76,
    },
    {
        "Model": "CNN + Transformer",
        "Family": "Hybrid",
        "RMSE (kW)": 12.9680,
        "MAE (kW)": 10.7033,
        "R²": 0.9147,
        "Accuracy %": 91.47,
    },
    {
        "Model": "Temporal Fusion Transformer",
        "Family": "Transformer",
        "RMSE (kW)": 18.5295,
        "MAE (kW)": 15.5136,
        "R²": 0.8258,
        "Accuracy %": 82.58,
    },
    {
        "Model": "CNN + LSTM",
        "Family": "Hybrid",
        "RMSE (kW)": 19.6084,
        "MAE (kW)": 16.5062,
        "R²": 0.8049,
        "Accuracy %": 80.49,
    },
]

frame = pd.DataFrame(data)

if METRICS_FILE.exists():
    payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    one_step = payload.get("one_step", [])
    if one_step:
        fresh = pd.DataFrame(one_step)
        fresh = fresh.rename(columns={"R2": "R²", "RMSE": "RMSE (kW)", "MAE": "MAE (kW)"})
        family_map = {
            "Random Forest": "Tree Ensemble",
            "PatchTST": "Transformer",
            "CNN + LSTM": "Hybrid",
            "CNN + Transformer": "Hybrid",
            "Temporal Fusion Transformer": "Transformer",
        }
        fresh["Family"] = fresh["Model"].map(family_map).fillna("Model")
        fresh["Accuracy %"] = fresh["R²"] * 100.0
        frame = fresh[["Model", "Family", "RMSE (kW)", "MAE (kW)", "R²", "Accuracy %"]]

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
