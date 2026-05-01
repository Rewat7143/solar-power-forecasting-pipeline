# ☀️ Solar Power Forecasting Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end machine learning pipeline for forecasting solar power generation using historical meter readings, NASA POWER weather data, and an ensemble of deep learning + tree-based models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project builds a production-style solar power forecasting system for a real-world solar installation (Warangal, Telangana, India — 17.94°N, 79.60°E). It ingests 5-minute resolution meter data from six solar inverters, merges it with satellite-derived weather features from the [NASA POWER API](https://power.larc.nasa.gov/), and trains multiple forecasting models to predict solar generation 24 hours ahead.

The pipeline supports both **one-step** (direct) and **recursive** (autoregressive) forecasting, includes a stacked ensemble meta-learner, and exports trained models for serving via a **Gradio UI** or a **Next.js web dashboard**.

---

## Key Features

- **Multi-source data ingestion** — Local Excel workbooks or Google Sheets with automatic tab merging
- **Automated weather alignment** — NASA POWER hourly → 5-min interpolation with fallback proxy
- **26 engineered features** — Cyclic time encodings, lag/rolling statistics, irradiance features
- **Nighttime anomaly sanitisation** — Automatic detection and zeroing of spurious night generation
- **5 forecasting models** — Random Forest, CNN+LSTM, PatchTST, CNN+Transformer, Temporal Fusion Transformer
- **Stacked ensemble** — Non-negative linear meta-learner over recursive backtest predictions
- **Affine calibration** — Post-hoc bias correction fitted on validation data
- **Rolling recursive backtests** — 24-hour rolling-origin evaluation with daylight-only metrics
- **Interactive prediction UIs** — Gradio desktop app + Next.js web dashboard
- **Model persistence** — joblib export with manifest.json for production serving

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│  Excel Workbooks (Inst_*.xlsx)  ◄──► Google Sheets (Gen_Sum)    │
│              ▼                                                  │
│     NASA POWER API (hourly weather) → 5-min interpolation       │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  Cyclic time features │ Lag/rolling stats │ Night sanitisation   │
│  26 features × daylight-only filtering → train/val/test split   │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                               │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Random Forest│  │ CNN+LSTM │  │  PatchTST   │  │CNN+Trans.│ │
│  └──────────────┘  └──────────┘  └─────────────┘  └──────────┘ │
│  ┌────────────────────────┐  ┌──────────────────────────────┐   │
│  │ Temporal Fusion Trans. │  │  Stacked Ensemble (meta-LR)  │   │
│  └────────────────────────┘  └──────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EVALUATION & EXPORT                             │
│  One-step metrics │ Recursive backtests │ Affine calibration     │
│  Model .pkl export │ manifest.json │ metrics JSON                │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING LAYER                                 │
│  Gradio UI (scripts/gradio_app.py)                              │
│  Next.js Dashboard (webapp/)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Machine Learning** | scikit-learn, Random Forest, LightGBM (optional) |
| **Deep Learning** | PyTorch (CNN+LSTM, PatchTST, TFT, CNN+Transformer) |
| **Weather Data** | NASA POWER REST API |
| **Visualisation** | Matplotlib, Seaborn |
| **Interactive UI** | Gradio |
| **Web Dashboard** | Next.js 14, TypeScript, Recharts |
| **Model Serving** | joblib, JSON manifest |

---

## Models

| Model | Type | One-Step R² | Recursive Daylight R² |
|---|---|---|---|
| **Random Forest** | Tree-based | 0.975 | 0.874 |
| **PatchTST** | Transformer | 0.928 | 0.700 |
| **CNN + Transformer** | Hybrid | 0.915 | -0.387 |
| **Temporal Fusion Transformer** | Attention | 0.826 | 0.218 |
| **CNN + LSTM** | Hybrid | 0.805 | -1.145 |
| **Stacked Ensemble** | Meta-learner | — | **0.875** |

> The stacked ensemble (RF + PatchTST + TFT + CNN+Transformer) achieves the best recursive daylight performance.

---

## Project Structure

```
solar-power-forecasting/
├── src/                            # Core Python package
│   ├── config.py                   #   Pipeline configuration & constants
│   ├── data_loader.py              #   Excel / Google Sheets ingestion
│   ├── weather.py                  #   NASA POWER API fetch & alignment
│   ├── features.py                 #   Feature engineering & splitting
│   ├── training.py                 #   DL training loop & CV search
│   ├── evaluation.py               #   Metrics, calibration, model export
│   ├── forecasting.py              #   Recursive forecast & backtesting
│   ├── visualization.py            #   Matplotlib plot helpers
│   └── models/
│       ├── sequence_models.py      #     Neural network architectures
│       └── ensemble.py             #     Stacked ensemble builder
├── scripts/
│   ├── run_pipeline.py             #   Main pipeline entry point
│   ├── gradio_app.py               #   Interactive prediction UI
│   ├── generate_report.py          #   HTML/PDF report generator
│   └── generate_visuals.py         #   Chart & diagram generators
├── notebooks/
│   └── solar_power_forecasting.ipynb  # Primary interactive notebook
├── data/                           # Raw meter Excel files (gitignored)
│   ├── 2025/{07..12}/
│   └── 2026/{01..04}/
├── models/                         # Trained model artifacts
│   ├── manifest.json
│   ├── Random_Forest/
│   ├── PatchTST/
│   ├── CNN_LSTM/
│   ├── CNN_Transformer/
│   └── Temporal_Fusion_Transformer/
├── outputs/
│   ├── images/                     # Generated analysis charts
│   ├── reports/                    # PDF/HTML/MD reports
│   └── metrics/                    # Latest metrics JSON
├── webapp/                         # Next.js web dashboard
│   ├── app/
│   ├── lib/
│   └── scripts/
├── google-apps-scripts/            # Google Apps Script integrations
├── docs/                           # Reference documents
├── .env.example                    # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (package manager)
- Node.js 18+ (only for the web dashboard)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/solar-power-forecasting.git
cd solar-power-forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Data Files

Place your Excel meter files in the `data/` directory following the date-based structure:

```
data/
├── 2025/
│   └── 07/
│       ├── Inst_2025-07-14.xlsx
│       └── ...
└── 2026/
    └── 01/
        └── ...
```

### 5. (Optional) Set Up the Web Dashboard

```bash
cd webapp
npm install
cp .env.example .env.local
cd ..
```

---

## Usage

### Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Load and merge solar meter data
2. Fetch NASA POWER weather features
3. Engineer 26 features
4. Train all models with cross-validation
5. Run recursive backtests
6. Build the stacked ensemble
7. Export models and metrics
8. Generate the 24-hour forecast

### Interactive Notebook

```bash
jupyter notebook notebooks/solar_power_forecasting.ipynb
```

### Gradio Prediction UI

```bash
python scripts/gradio_app.py
```

Opens a local web UI at `http://127.0.0.1:7860` for interactive predictions.

### Next.js Web Dashboard

```bash
cd webapp
npm run dev
```

Opens the dashboard at `http://localhost:3000` with live prediction charts and confidence bands.

---

## Results

### One-Step Prediction (Test Set)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Random Forest | 3.34 | 6.98 | 0.975 |
| PatchTST | 7.49 | 11.94 | 0.928 |
| CNN + Transformer | 10.70 | 12.97 | 0.915 |
| Temporal Fusion Transformer | 15.51 | 18.53 | 0.826 |
| CNN + LSTM | 16.51 | 19.61 | 0.805 |

### Recursive 24-Hour Backtest (Daylight Only)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| **Stacked Ensemble** | **10.18** | **15.69** | **0.875** |
| Random Forest | 10.50 | 15.78 | 0.874 |
| PatchTST | 17.00 | 24.32 | 0.700 |

---

## Screenshots

> Generated analysis charts are located in `outputs/images/`.

| Chart | Description |
|---|---|
| `1_actual_vs_predicted_all_models.png` | All models compared against actual generation |
| `2_correlation_heatmap.png` | Feature correlation matrix |
| `3_future_forecast_next_24h.png` | 24-hour ahead forecast curve |
| `4_model_comparison_table.png` | Tabular model performance summary |
| `5_model_metrics_bar_chart.png` | Visual metrics comparison |
| `6_raw_time_series_plot.png` | Raw solar generation time series |
| `7_recursive_daylight_metrics_bar_chart.png` | Recursive backtest metrics |
| `8_residual_plot_error_analysis.png` | Residual analysis by model |
| `9_system_architecture_diagram.png` | Pipeline architecture diagram |

---

## Future Improvements

- [ ] **Real-time weather API** — Replace historical proxy with live weather forecast (OpenWeatherMap / Tomorrow.io)
- [ ] **Probabilistic forecasting** — Quantile regression or conformal prediction for calibrated uncertainty bands
- [ ] **Hyperparameter optimisation** — Automated tuning with Optuna or Ray Tune
- [ ] **Cloud deployment** — Dockerise the pipeline and deploy the dashboard on Vercel / AWS
- [ ] **Multi-site support** — Generalise for multiple solar installations
- [ ] **Battery storage integration** — Extend to solar + storage optimisation
- [ ] **Anomaly detection** — Automated detection of inverter faults and degradation

---

## Author

**Rewat** — Solar Power Forecasting Pipeline

- GitHub: [@Rewat7143](https://github.com/Rewat7143)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
