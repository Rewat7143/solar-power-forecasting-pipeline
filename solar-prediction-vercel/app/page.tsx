"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type PredictResponse = {
  ok: boolean;
  generatedAt: string;
  requestDate: string;
  requestTime: string;
  resolvedModel?: string;
  dataSource?: string;
  note: string;
  summary: {
    currentSolarKw: number;
    peakSolarKw: number;
    avgSolarKw: number;
    confidenceBandKw: { low: number; high: number };
    status: "active" | "standby";
  };
  chartSeries: {
    labels: string[];
    solarPowerKw: number[];
    lowBandKw: number[];
    highBandKw: number[];
    selectedIdx: number;
    meta: Array<{
      timestamp: string;
      irradiance_wm2: number;
      cloud_pct: number;
      temp_c: number;
      status: string;
      slot_mean_kw: number;
      slot_std_kw: number;
    }>;
  };
  forecastColumns: Array<{
    timestamp: string;
    timeLabel: string;
    solar_power_kw: number;
    confidence_low_kw: number;
    confidence_high_kw: number;
    irradiance_proxy: number;
    cloud_pct: number;
    temp_c: number;
    status: string;
    source: string;
  }>;
};

function num(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default function HomePage() {
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<PredictResponse | null>(null);
  const didInitialPredict = useRef(false);

  const predict = useCallback(async (requestDate?: string, requestTime?: string) => {
    const effectiveDate = typeof requestDate === "string" ? requestDate : date;
    const effectiveTime = typeof requestTime === "string" ? requestTime : time;
    if (!effectiveDate || !effectiveTime) {
      setError("Select both date and time before predicting.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: effectiveDate, time: effectiveTime }),
      });
      const raw = await response.text();
      let json: (PredictResponse & { error?: string }) | null = null;
      try {
        json = JSON.parse(raw) as PredictResponse & { error?: string };
      } catch {
        throw new Error(raw || "Prediction failed");
      }
      if (!response.ok || !json.ok) {
        throw new Error(json.error || raw || "Prediction failed");
      }
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [date, time]);

  useEffect(() => {
    if (didInitialPredict.current) return;
    didInitialPredict.current = true;
    // Default to a date with good solar generation from training data
    const defaultDate = "2025-07-15";
    const defaultTime = "13:00";
    setDate(defaultDate);
    setTime(defaultTime);
    void predict(defaultDate, defaultTime);
  }, [predict]);

  const chartData = useMemo(() => {
    if (!data?.chartSeries) return [];
    const s = data.chartSeries;
    return s.labels.map((label, i) => ({
      label,
      solar: num(s.solarPowerKw[i]),
      low: num(s.lowBandKw[i]),
      high: num(s.highBandKw[i]),
      meta: s.meta[i],
      selected: i === s.selectedIdx,
    }));
  }, [data]);

  const summary = data?.summary;
  const sourceLabel = data?.resolvedModel ?? data?.forecastColumns?.[0]?.source ?? "unknown";
  const hasPrediction = Boolean(summary);
  const fmtValue = (value: unknown, decimals = 2, unit = "kW") =>
    hasPrediction ? `${num(value).toFixed(decimals)} ${unit}` : "--";

  return (
    <main className="page-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <div className="page-frame">
        <header className="hero">
          <div className="hero-copy">
            <div className="eyebrow">Solar prediction workspace</div>
            <h1>Real model predictions, rendered in a sharper live dashboard.</h1>
            <p>
              Select a date and time, run the trained Random Forest model, and inspect the selected timestamp with
              confidence bands, weather context, and the full daily curve.
            </p>
            <div className="hero-badges">
              <span className="badge badge-live">{loading ? "Refreshing live model" : "Live model ready"}</span>
              <span className="badge">{data?.requestDate ?? date}</span>
              <span className="badge">{data?.requestTime ?? time}</span>
            </div>
          </div>

          <div className="hero-panel">
            <div className="hero-panel-top">
              <div>
                <span className="panel-label">Selected timestamp</span>
                <h2>{summary ? `${data?.requestDate} ${data?.requestTime}` : `${date} ${time}`}</h2>
              </div>
              <span className={`signal ${summary?.status === "active" ? "signal-on" : "signal-off"}`}>
                {summary?.status ?? "standby"}
              </span>
            </div>
            <div className="hero-panel-grid">
              <div>
                <span className="panel-label">Current solar</span>
                <strong>{fmtValue(summary?.currentSolarKw)}</strong>
              </div>
              <div>
                <span className="panel-label">Confidence</span>
                <strong>{hasPrediction ? `${num(summary?.confidenceBandKw?.low).toFixed(1)} to ${num(summary?.confidenceBandKw?.high).toFixed(1)}` : "--"}</strong>
              </div>
              <div>
                <span className="panel-label">Peak today</span>
                <strong>{fmtValue(summary?.peakSolarKw)}</strong>
              </div>
              <div>
                <span className="panel-label">Average today</span>
                <strong>{fmtValue(summary?.avgSolarKw)}</strong>
              </div>
              <div>
                <span className="panel-label">Model source</span>
                <strong>{sourceLabel}</strong>
              </div>
            </div>
            <div className="hero-panel-footer">
              <div>
                <span className="panel-label">Model note</span>
                <p>{data?.note ?? "Prediction is for a single selected timestamp, not a 24-hour future forecast."}</p>
              </div>
              <button className="btn-primary" onClick={() => void predict()} disabled={loading}>
                {loading ? "Predicting..." : "Refresh prediction"}
              </button>
            </div>
          </div>
        </header>

        <section className="controls-row">
          <div className="control-card">
            <div>
              <span className="panel-label">Choose date and time</span>
              <p>Use the current timestamp or move to a historical slot for comparison.</p>
            </div>
            <div className="tools">
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
              <input type="time" step={300} value={time} onChange={(e) => setTime(e.target.value)} />
              <button className="btn-primary" onClick={() => void predict()} disabled={loading}>
                Predict
              </button>
            </div>
          </div>
          <div className="status-card">
            <span className="panel-label">System status</span>
            <strong>{loading ? "Running inference" : "Idle / ready"}</strong>
            <p>{data?.generatedAt ? `Updated ${data.generatedAt}` : "Waiting for prediction"}</p>
            <p>{sourceLabel !== "unknown" ? `Connected to ${sourceLabel}` : "Source: unknown"}</p>
          </div>
        </section>

        {error ? <div className="error">{error}</div> : null}

        <section className="metrics-grid">
          <Card title="Current Solar" value={summary?.currentSolarKw} unit="kW" status={summary?.status ?? "standby"} accent="amber" />
          <Card title="Peak (Today)" value={summary?.peakSolarKw} unit="kW" status="prediction" accent="green" />
          <Card title="Average (Today)" value={summary?.avgSolarKw} unit="kW" status="trend" accent="blue" />
          <Card
            title="Confidence Band"
            valueText={
              summary
                ? `${num(summary?.confidenceBandKw?.low).toFixed(1)} - ${num(summary?.confidenceBandKw?.high).toFixed(1)} kW`
                : "--"
            }
            status="range"
            accent="slate"
          />
        </section>

        <section className="chart-card">
          <div className="section-head">
            <div>
              <span className="panel-label">Forecast curve</span>
              <h2>24-hour context around the chosen timestamp</h2>
            </div>
            <div className="legend-inline">
              <span><i className="dot dot-gold" />Solar</span>
              <span><i className="dot dot-green" />Upper band</span>
              <span><i className="dot dot-slate" />Lower band</span>
            </div>
          </div>

          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={380}>
              <AreaChart data={chartData} margin={{ top: 16, right: 18, bottom: 6, left: 0 }}>
                <defs>
                  <linearGradient id="solarFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.42} />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.04} />
                  </linearGradient>
                  <linearGradient id="highFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#16a34a" stopOpacity={0.24} />
                    <stop offset="100%" stopColor="#16a34a" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="lowFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#64748b" stopOpacity={0.18} />
                    <stop offset="100%" stopColor="#64748b" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(148,163,184,0.18)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 12 }} tickMargin={12} axisLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickMargin={10} width={44} />
                <Tooltip
                  cursor={{ stroke: "rgba(245,158,11,0.28)", strokeWidth: 1 }}
                  formatter={(value: number, name: string) => [`${num(value).toFixed(2)} kW`, name]}
                  labelFormatter={(label, payload) => {
                    const m = payload?.[0]?.payload?.meta;
                    if (!m) return `Time: ${label}`;
                    return `Time ${label} | Irr ${num(m.irradiance_wm2).toFixed(1)} W/m² | Cloud ${num(m.cloud_pct).toFixed(1)}% | Temp ${num(m.temp_c).toFixed(1)}°C`;
                  }}
                />
                <Area type="monotone" dataKey="high" stroke="#16a34a" fill="url(#highFill)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="low" stroke="#64748b" fill="url(#lowFill)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="solar" stroke="#f59e0b" fill="url(#solarFill)" strokeWidth={3} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="table-card">
          <div className="section-head">
            <div>
              <span className="panel-label">Prediction detail</span>
              <h2>Selected timestamp and neighboring context</h2>
            </div>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Time</th>
                  <th>Solar (kW)</th>
                  <th>Low (kW)</th>
                  <th>High (kW)</th>
                  <th>Irradiance</th>
                  <th>Cloud %</th>
                  <th>Temp C</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(data?.forecastColumns ?? []).map((r) => (
                  <tr key={r.timestamp} className={r.timestamp === data?.forecastColumns?.[0]?.timestamp ? "row-highlight" : ""}>
                    <td className="mono">{r.timestamp}</td>
                    <td className="mono">{r.timeLabel}</td>
                    <td className="mono">{num(r.solar_power_kw).toFixed(2)}</td>
                    <td className="mono">{num(r.confidence_low_kw).toFixed(2)}</td>
                    <td className="mono">{num(r.confidence_high_kw).toFixed(2)}</td>
                    <td className="mono">{num(r.irradiance_proxy).toFixed(1)}</td>
                    <td className="mono">{num(r.cloud_pct).toFixed(1)}</td>
                    <td className="mono">{num(r.temp_c).toFixed(1)}</td>
                    <td><span className={`status-pill ${r.status === "active" ? "status-pill-on" : "status-pill-off"}`}>{r.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}

function Card({
  title,
  value,
  unit,
  valueText,
  status,
  accent,
}: {
  title: string;
  value?: number;
  unit?: string;
  valueText?: string;
  status: string;
  accent: "amber" | "green" | "blue" | "slate";
}) {
  return (
    <div className={`card metric-card metric-${accent}`}>
      <div className="card-topline">
        <div className="label">{title}</div>
        <span className={`status-chip ${status === "active" ? "active" : status === "standby" ? "standby" : "neutral"}`}>{status}</span>
      </div>
      <div className="value mono">{valueText ?? (typeof value === "number" ? `${num(value).toFixed(2)} ${unit ?? ""}` : "--")}</div>
      <div className="card-foot">Selected model output for the chosen timestamp</div>
    </div>
  );
}
