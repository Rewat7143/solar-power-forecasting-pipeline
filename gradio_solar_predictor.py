#!/usr/bin/env python3
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICT_SCRIPT = ROOT / "solar-prediction-vercel" / "scripts" / "predict_with_model.py"
MANIFEST_PATH = ROOT / "models" / "manifest.json"
DATA_ROOT = ROOT
AUTO_MODEL_OPTION = "Auto (Test all models)"


def load_model_choices() -> list[str]:
    models = ["Random Forest"]
    if not MANIFEST_PATH.exists():
        return [AUTO_MODEL_OPTION] + models
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        saved_models = manifest.get("saved_models", {})
        if isinstance(saved_models, dict) and saved_models:
            models = list(saved_models.keys())
    except Exception:
        pass
    return [AUTO_MODEL_OPTION] + models


def run_predict_script(date_str: str, time_str: str, model_name: str):
    payload = {
        "date": date_str,
        "time": time_str,
        "manifestPath": str(MANIFEST_PATH),
        "dataRoot": str(DATA_ROOT),
        "modelName": model_name,
    }

    proc = subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"Prediction script error (code {proc.returncode}): {stderr}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Prediction script returned invalid JSON output.") from exc


def find_metric_record(result: dict, resolved_model: str) -> dict:
    metrics = result.get("selectedMetrics", [])
    for record in metrics:
        model_label = str(record.get("Model") or record.get("model") or "").strip()
        if model_label == resolved_model:
            return record
    return {}


def metric_as_float(record: dict, keys: list[str], default: float) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except Exception:
                continue
    return default


def choose_best_model_row(rows: list[dict]) -> dict:
    # Rank by validation metrics first (lower RMSE/MAE and higher R2), then by higher current kW.
    return min(
        rows,
        key=lambda row: (
            row["rmse"],
            row["mae"],
            -row["r2"],
            -row["current_kw"],
        ),
    )


def run_prediction(date_str: str, time_str: str, model_name: str):
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    if not date_str or not time_str:
        return "Provide both date and time.", "", pd.DataFrame(), pd.DataFrame()

    if not PREDICT_SCRIPT.exists():
        return f"Prediction script not found: {PREDICT_SCRIPT}", "", pd.DataFrame(), pd.DataFrame()

    try:
        if model_name == AUTO_MODEL_OPTION:
            candidate_models = [m for m in load_model_choices() if m != AUTO_MODEL_OPTION]
            tested_rows = []
            tested_results = {}

            for candidate in candidate_models:
                result = run_predict_script(date_str, time_str, candidate)
                resolved = str(result.get("resolvedModel", candidate))
                summary = result.get("summary", {})
                metric_row = find_metric_record(result, resolved)

                tested_rows.append(
                    {
                        "candidate": candidate,
                        "resolved_model": resolved,
                        "current_kw": float(summary.get("currentSolarKw", 0.0)),
                        "peak_kw": float(summary.get("peakSolarKw", 0.0)),
                        "avg_kw": float(summary.get("avgSolarKw", 0.0)),
                        "rmse": metric_as_float(metric_row, ["RMSE", "rmse"], float("inf")),
                        "mae": metric_as_float(metric_row, ["MAE", "mae"], float("inf")),
                        "r2": metric_as_float(metric_row, ["R2", "r2"], float("-inf")),
                        "data_source": str(result.get("dataSource", "unknown")),
                    }
                )
                tested_results[resolved] = result

            if not tested_rows:
                return "No models available to test.", "", pd.DataFrame(), pd.DataFrame()

            best_row = choose_best_model_row(tested_rows)
            result = tested_results[best_row["resolved_model"]]
            comparison_df = pd.DataFrame(tested_rows).sort_values(
                by=["rmse", "mae", "r2"],
                ascending=[True, True, False],
            )
            auto_note = (
                f"Auto mode tested {len(tested_rows)} models. "
                f"Best by validation metrics: {best_row['resolved_model']} "
                f"(RMSE={best_row['rmse']:.3f}, MAE={best_row['mae']:.3f}, R2={best_row['r2']:.3f})."
            )
        else:
            result = run_predict_script(date_str, time_str, model_name)
            comparison_df = pd.DataFrame()
            auto_note = ""
    except subprocess.TimeoutExpired:
        return "Prediction timed out after 240 seconds.", "", pd.DataFrame(), pd.DataFrame()
    except Exception as exc:
        return f"Prediction failed: {exc}", "", pd.DataFrame(), pd.DataFrame()

    summary = result.get("summary", {})
    ds = result.get("dataSource", "unknown")
    resolved = result.get("resolvedModel", model_name)
    status_text = (
        f"Model: {resolved}\n"
        f"Data Source: {ds}\n"
        f"Current kW: {summary.get('currentSolarKw', 'n/a')}\n"
        f"Peak kW: {summary.get('peakSolarKw', 'n/a')}\n"
        f"Average kW: {summary.get('avgSolarKw', 'n/a')}"
    )
    if auto_note:
        status_text = f"{status_text}\n{auto_note}"

    chart = result.get("chartSeries", {})
    df = pd.DataFrame(
        {
            "time": chart.get("labels", []),
            "solar_kw": chart.get("solarPowerKw", []),
            "low_kw": chart.get("lowBandKw", []),
            "high_kw": chart.get("highBandKw", []),
        }
    )

    raw_json = json.dumps(result, indent=2)
    return status_text, raw_json, df, comparison_df


def default_date_time() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M")


def build_app() -> gr.Blocks:
    dflt_date, dflt_time = default_date_time()
    model_choices = load_model_choices()

    with gr.Blocks(title="Solar Generation kW Predictor") as app:
        gr.Markdown("# Solar Generation kW Predictor")
        gr.Markdown("Run model prediction directly from your local training outputs (without the web app).")

        with gr.Row():
            date_input = gr.Textbox(label="Date (YYYY-MM-DD)", value=dflt_date)
            time_input = gr.Textbox(label="Time (HH:MM)", value=dflt_time)
            model_input = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0],
                label="Model",
                interactive=True,
            )

        predict_btn = gr.Button("Predict")

        status_output = gr.Textbox(label="Summary", lines=6)
        raw_output = gr.Code(label="Raw JSON Response", language="json")
        table_output = gr.Dataframe(label="24h Prediction Series", interactive=False)
        model_test_output = gr.Dataframe(label="All-Model Test Results (Auto Mode)", interactive=False)

        predict_btn.click(
            fn=run_prediction,
            inputs=[date_input, time_input, model_input],
            outputs=[status_output, raw_output, table_output, model_test_output],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
