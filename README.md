# Solar Power Forecasting Pipeline

End-to-end solar generation forecasting project using historical solar meter readings and NASA weather data.

## What This Project Does

- Ingests solar meter data from local Excel exports (`MTR_24` to `MTR_29`, `Gen_Sum`).
- Pulls and aligns NASA POWER weather features:
  - `ALLSKY_SFC_SW_DWN` (irradiance)
  - `T2M` (temperature)
  - `CLOUD_AMT` (cloud cover)
- Performs preprocessing, feature engineering, model training, and evaluation.
- Builds multiple forecasting models and an ensemble.
- Produces visual outputs and an interactive Gradio forecast UI.

## Models Included

- Random Forest
- PatchTST
- CNN + LSTM
- CNN + Transformer
- Temporal Fusion Transformer (TFT)
- Ensemble (RF + PatchTST)

## Main Files

- `solar_power_forecasting_pipeline.ipynb` — primary notebook workflow
- `solar_power_forecasting_pipeline.py` — script version of the pipeline
- `models/` — exported trained models + `manifest.json`
- `output_images_v3/` — generated analysis and report images
- `MODEL_REPORT.md` — full report draft
- `QUICK_REFERENCE.md` — short summary report
- `report.html` — formatted project report

## Generated Visual Artifacts

Inside `output_images_v3/`:

- `actual_vs_predicted_all_models.png`
- `future_forecast_next_24h.png`
- `model_comparison_table.png`
- `model_metrics_bar_chart.png`
- `residual_plot_error_analysis.png`
- `raw_time_series_plot.png`
- `correlation_heatmap.png`
- `system_architecture_diagram.png`

## Setup

Python 3.11+ recommended.

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib requests torch openpyxl nbformat lightgbm gradio seaborn
```

## Run

### Notebook workflow

```bash
jupyter notebook solar_power_forecasting_pipeline.ipynb
```

Run cells sequentially.

### Generate report visuals/scripts

```bash
python3 generate_html_report.py
python3 generate_model_comparison_image.py
python3 generate_model_metrics_bar_chart.py
python3 generate_system_architecture_diagram.py
```

## Forecast Output

- One-step and recursive forecasts
- Next 24-hour forecast with timestamped output
- Saved visual diagnostics for error analysis and model comparison

## Notes

- LightGBM is installed but intentionally disabled in this environment for Python 3.14 kernel stability.
- Forecast UI is available through Gradio in the notebook.

## Project Goal

Provide a reproducible, research-grade solar forecasting pipeline that combines physical weather drivers and time-series modeling for practical prediction and reporting.
