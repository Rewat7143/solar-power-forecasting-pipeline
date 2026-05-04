"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toPng } from "html-to-image";

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
    currentObservedKw?: number | null;
    peakSolarKw: number;
    avgSolarKw: number;
    confidenceBandKw: { low: number; high: number };
    status: "active" | "standby";
  };
  chartSeries: {
    labels: string[];
    solarPowerKw: number[];
    observedPowerKw?: (number | null)[];
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
    observed_power_kw?: number | null;
    confidence_low_kw: number;
    confidence_high_kw: number;
    irradiance_proxy: number;
    cloud_pct: number;
    temp_c: number;
    status: string;
    source: string;
  }>;
};

const MODELS = [
  "Random Forest",
  "CNN + LSTM",
  "PatchTST",
  "Temporal Fusion Transformer",
  "CNN + Transformer",
  "Stacked Ensemble (Random Forest + PatchTST + Temporal Fusion Transformer + CNN + Transformer)"
];

function num(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default function DashboardPage() {
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [selectedModel, setSelectedModel] = useState("Random Forest");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<PredictResponse | null>(null);
  const didInitialPredict = useRef(false);
  const chartRef = useRef<HTMLDivElement>(null);

  const handleDownloadImage = async () => {
    if (!chartRef.current) return;
    try {
      const dataUrl = await toPng(chartRef.current, { backgroundColor: '#ffffff', cacheBust: true });
      const link = document.createElement('a');
      link.download = `solar-forecast-chart-${data?.requestDate || 'export'}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to export chart:', err);
    }
  };

  const handleDownloadCSV = () => {
    if (!data || !chartData.length) return;
    
    const rows = [
      ["Time", "Predicted Solar (kW)", "Observed Solar (kW)", "Lower Confidence (kW)", "Upper Confidence (kW)", "Irradiance (W/m2)", "Cloud Coverage (%)", "Temperature (C)", "System Status"],
      ...chartData.map(d => [
        d.label,
        d.solar.toFixed(3),
        d.observed !== null ? d.observed.toFixed(3) : "N/A",
        d.low.toFixed(3),
        d.high.toFixed(3),
        num(d.meta.irradiance_wm2).toFixed(1),
        num(d.meta.cloud_pct).toFixed(1),
        num(d.meta.temp_c).toFixed(1),
        d.meta.status
      ])
    ];

    const csvContent = rows.map(r => r.map(cell => `"${cell}"`).join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `solar_forecast_24h_${data.requestDate}_${data.requestTime.replace(':', '-')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const predict = useCallback(async (requestDate?: string, requestTime?: string, reqModel?: string) => {
    const effectiveDate = typeof requestDate === "string" ? requestDate : date;
    const effectiveTime = typeof requestTime === "string" ? requestTime : time;
    const effectiveModel = typeof reqModel === "string" ? reqModel : selectedModel;
    
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
        body: JSON.stringify({ date: effectiveDate, time: effectiveTime, model: effectiveModel }),
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
  }, [date, time, selectedModel]);

  useEffect(() => {
    if (didInitialPredict.current) return;
    didInitialPredict.current = true;
    
    // Get current system time in local timezone
    const now = new Date();
    
    // Format YYYY-MM-DD
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    const defaultDate = `${yyyy}-${mm}-${dd}`;
    
    // Format HH:MM (24-hour)
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const defaultTime = `${hours}:${minutes}`;
    
    setDate(defaultDate);
    setTime(defaultTime);
    // Removed automatic predict() call so user has to click explicitly
  }, []);

  const chartData = useMemo(() => {
    if (!data?.chartSeries) return [];
    const s = data.chartSeries;
    return s.labels.map((label, i) => ({
      label,
      solar: num(s.solarPowerKw[i]),
      observed: s.observedPowerKw?.[i] ?? null,
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

  // Financial & Environmental Calcs
  const TARIFF_RATE = 8.00; // INR per kWh
  const CO2_FACTOR = 0.82; // kg CO2 per kWh
  const dailyEnergyKwh = summary ? summary.avgSolarKw * 24 : 0;
  const dailySavings = dailyEnergyKwh * TARIFF_RATE;
  const co2Offset = dailyEnergyKwh * CO2_FACTOR;

  // System Health / Anomaly Detection
  let healthStatus = "none";
  let healthMsg = "Awaiting inference...";
  
  if (summary && summary.status === "active") {
    if (summary.currentObservedKw === null || summary.currentObservedKw === undefined) {
      healthStatus = "info";
      healthMsg = "Forecast Mode: Waiting for live physical telemetry synchronization for this timestamp.";
    } else if (summary.currentObservedKw <= 0) {
      healthStatus = "critical";
      healthMsg = "CRITICAL: Inverter Down. Telemetry reports 0 kW during daylight hours.";
    } else if (summary.currentObservedKw < summary.confidenceBandKw.low - 5) {
      // Small 5kW grace period on the lower bound
      healthStatus = "warning";
      healthMsg = `WARNING: Live generation (${summary.currentObservedKw.toFixed(1)} kW) is critically below the 95% confidence bounds. Potential shading, dust accumulation, or partial inverter failure.`;
    } else {
      healthStatus = "optimal";
      healthMsg = "Optimal System Health. Physical plant is matching or exceeding neural network predictions.";
    }
  } else if (summary && summary.status === "standby") {
    healthStatus = "standby";
    healthMsg = "System on Night Standby (Irradiance too low).";
  }

  return (
    <div className="page-frame">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4">
        <div>
          <span className="eyebrow">System Interface</span>
          <h1 className="text-3xl font-bold text-[#1E293B]">Live Predictor Dashboard</h1>
        </div>
        <div className="flex items-center gap-2 bg-white px-4 py-2 rounded-md border border-slate-200 shadow-sm">
          <span className={`w-2.5 h-2.5 rounded-full ${loading ? "bg-[#D97706] animate-pulse" : "bg-green-600"}`}></span>
          <span className="text-sm text-slate-600 font-semibold tracking-wide uppercase">
            {loading ? (
              <span className="flex items-center gap-2">
                Computing Inference... 
                <span className="text-[10px] text-slate-400 normal-case font-normal">(Initial startup may take 50s on free tier)</span>
              </span>
            ) : "System Active"}
          </span>
        </div>
      </div>

      {summary && (
        <div className={`mb-8 p-4 rounded-lg border-l-4 flex items-start gap-4 shadow-sm ${
          healthStatus === 'optimal' ? 'bg-green-50 border-green-500' :
          healthStatus === 'warning' ? 'bg-amber-50 border-amber-500' :
          healthStatus === 'critical' ? 'bg-red-50 border-red-500' :
          healthStatus === 'info' ? 'bg-blue-50 border-blue-500' :
          'bg-slate-50 border-slate-400'
        }`}>
          <div className="mt-0.5">
            {healthStatus === 'optimal' && <span className="text-green-600 text-xl">✅</span>}
            {healthStatus === 'warning' && <span className="text-amber-600 text-xl">⚠️</span>}
            {healthStatus === 'critical' && <span className="text-red-600 text-xl">🚨</span>}
            {healthStatus === 'info' && <span className="text-blue-600 text-xl">📡</span>}
            {healthStatus === 'standby' && <span className="text-slate-600 text-xl">🌙</span>}
          </div>
          <div>
            <h3 className={`font-bold text-lg ${
              healthStatus === 'optimal' ? 'text-green-800' :
              healthStatus === 'warning' ? 'text-amber-800' :
              healthStatus === 'critical' ? 'text-red-800' :
              healthStatus === 'info' ? 'text-blue-800' :
              'text-slate-800'
            }`}>
              {healthStatus === 'optimal' ? 'System Health: Optimal' :
               healthStatus === 'warning' ? 'System Health: Underperforming' :
               healthStatus === 'critical' ? 'System Health: CRITICAL' :
               healthStatus === 'info' ? 'System Health: Awaiting Telemetry' :
               'System Health: Standby'}
            </h3>
            <p className={`mt-1 text-sm ${
              healthStatus === 'optimal' ? 'text-green-700' :
              healthStatus === 'warning' ? 'text-amber-700' :
              healthStatus === 'critical' ? 'text-red-700' :
              healthStatus === 'info' ? 'text-blue-700' :
              'text-slate-600'
            }`}>
              {healthMsg}
            </p>
          </div>
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6 mb-8">
        <div className="card-panel lg:col-span-2">
          <h2 className="text-xl font-bold mb-6 text-[#1E293B]">Simulation Parameters</h2>
          <div className="grid sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-slate-600 mb-2 font-semibold">Observation Date</label>
              <input type="date" className="w-full" value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-2 font-semibold">Time Interval</label>
              <input type="time" step={300} className="w-full" value={time} onChange={(e) => setTime(e.target.value)} />
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-2 font-semibold">Inference Model</label>
              <select className="w-full" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {MODELS.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-6 flex gap-4">
            <button className="btn-primary" onClick={() => void predict()} disabled={loading}>
              {loading ? "Processing..." : "Execute Prediction"}
            </button>
            {data && (
              <button 
                className="btn-secondary" 
                onClick={handleDownloadCSV}
              >
                Download CSV
              </button>
            )}
          </div>
          {error && <div className="mt-4 error-message">{error}</div>}
        </div>

        <div className="card-panel">
          <h2 className="text-xl font-bold mb-6 text-[#1E293B]">Model Metadata</h2>
          
          {data && (
          <div className="card-panel border-t-4 border-[#10B981]">
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-slate-500 font-semibold text-sm">Target Status</h3>
              <span className={`text-xs px-2 py-1 rounded-full uppercase tracking-wider font-bold ${
                data.summary.status === "active" ? "bg-green-100 text-green-700" : "bg-slate-200 text-slate-600"
              }`}>
                {data.summary.status}
              </span>
            </div>
            {data.summary.currentObservedKw !== null && data.summary.currentObservedKw !== undefined ? (
              <>
                <div className="text-2xl font-bold text-[#10B981]">{data.summary.currentObservedKw} kW</div>
                <div className="text-sm text-slate-500 font-medium uppercase mt-1">Live Observed</div>
                <div className="text-lg font-bold text-slate-700 mt-2">{data.summary.currentSolarKw} kW</div>
                <div className="text-xs text-slate-400 font-medium uppercase">Predicted</div>
              </>
            ) : (
              <>
                <div className="text-3xl font-bold text-[#1E293B]">{data.summary.currentSolarKw} kW</div>
                <div className="text-sm text-[#D97706] font-medium uppercase mt-1">Predicted Output</div>
              </>
            )}
          </div>
          )}

          <div className="space-y-4">
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Selected Source</div>
              <div className="font-semibold text-[#1E3A8A] mt-1">{sourceLabel}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Data Pipeline</div>
              <div className="font-medium text-slate-800 mt-1">{data?.dataSource || "--"}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold">System Note</div>
              <div className="text-sm mt-1 text-slate-600 bg-slate-50 p-3 rounded border border-slate-100">{data?.note || "Awaiting prediction."}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-8">
        <MetricCard title="Current Solar" value={summary?.currentSolarKw} unit="kW" status={summary?.status ?? "standby"} accent="amber" />
        <MetricCard title="Peak (Today)" value={summary?.peakSolarKw} unit="kW" status="prediction" accent="green" />
        <MetricCard title="Average (Today)" value={summary?.avgSolarKw} unit="kW" status="trend" accent="blue" />
        <MetricCard 
          title="95% Confidence" 
          valueText={summary ? `${num(summary?.confidenceBandKw?.low).toFixed(1)} - ${num(summary?.confidenceBandKw?.high).toFixed(1)}` : "--"} 
          status="range" 
          accent="slate" 
        />
        <MetricCard 
          title="Est. Savings (Today)" 
          valueText={summary ? `₹${dailySavings.toFixed(0)}` : "--"} 
          status="financial" 
          accent="green" 
        />
        <MetricCard 
          title="CO₂ Offset (Today)" 
          valueText={summary ? `${co2Offset.toFixed(1)} kg` : "--"} 
          status="environmental" 
          accent="blue" 
        />
      </div>

      <div className="card-panel mb-8">
        <div className="flex justify-between items-end mb-6 flex-wrap gap-4">
          <div>
            <h2 className="text-xl font-bold text-[#1E293B]">Diurnal Forecast Curve (24h)</h2>
            <p className="text-slate-500 text-sm mt-1">Contextual generation profile for the evaluated day</p>
          </div>
          <div className="flex flex-wrap gap-3 items-center">
            {data && (
              <div className="flex gap-2 mr-4 border-r border-slate-200 pr-4">
                <button 
                  onClick={handleDownloadCSV}
                  className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded text-xs font-bold text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
                  title="Download telemetry as CSV"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                  Export CSV
                </button>
                <button 
                  onClick={handleDownloadImage}
                  className="flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 rounded text-xs font-bold text-[#1E3A8A] hover:bg-slate-50 transition-colors shadow-sm"
                  title="Save chart as PNG"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                  Save Image
                </button>
              </div>
            )}
            <div className="flex flex-wrap gap-4 text-[11px] md:text-xs text-slate-600 font-bold uppercase tracking-tight">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#10B981]"></span> Observed</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#D97706]"></span> Predicted</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#3B82F6]"></span> Confidence</span>
            </div>
          </div>
        </div>
        
        <div ref={chartRef} className="h-[300px] md:h-[400px] w-full bg-slate-50 rounded-lg p-2 md:p-4 border border-slate-200">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 16, right: 18, bottom: 6, left: 0 }}>
              <defs>
                <linearGradient id="solarFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#D97706" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#D97706" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="highFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="lowFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#94A3B8" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#94A3B8" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="4 4" stroke="#E2E8F0" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: "#64748B", fontSize: 12, fontWeight: 500 }} tickMargin={12} axisLine={false} />
              <YAxis tick={{ fill: "#64748B", fontSize: 12, fontWeight: 500 }} axisLine={false} tickMargin={10} width={44} />
              <Tooltip
                contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#E2E8F0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                itemStyle={{ color: '#1E293B', fontWeight: 600 }}
                cursor={{ stroke: "#D97706", strokeWidth: 1, strokeDasharray: "4 4" }}
                formatter={(value: number, name: string) => {
                  const displayName = name === 'solar' ? 'Predicted' : name === 'observed' ? 'Live Observed' : name;
                  return [`${num(value).toFixed(2)} kW`, displayName];
                }}
                labelFormatter={(label, payload) => {
                  const m = payload?.[0]?.payload?.meta;
                  if (!m) return `Time: ${label}`;
                  return `Time ${label} | Irr ${num(m.irradiance_wm2).toFixed(1)} W/m² | Cloud ${num(m.cloud_pct).toFixed(1)}% | Temp ${num(m.temp_c).toFixed(1)}°C`;
                }}
              />
              <Area type="monotone" dataKey="high" stroke="#3B82F6" fill="url(#highFill)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="low" stroke="#94A3B8" fill="url(#lowFill)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="solar" stroke="#D97706" fill="url(#solarFill)" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="observed" stroke="#10B981" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card-panel">
        <h2 className="text-xl font-bold mb-6 text-[#1E293B]">Point-in-Time Telemetry</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Solar (kW)</th>
                <th>Low (kW)</th>
                <th>High (kW)</th>
                <th>Irradiance (W/m²)</th>
                <th>Cloud (%)</th>
                <th>Temp (°C)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(data?.forecastColumns ?? []).map((r) => (
                <tr key={r.timestamp} className={r.timestamp === data?.forecastColumns?.[0]?.timestamp ? "row-highlight" : ""}>
                  <td className="mono font-semibold text-slate-700">{r.timeLabel}</td>
                  <td className="mono font-semibold text-slate-800">{num(r.solar_power_kw).toFixed(2)}</td>
                  <td className="mono text-slate-500">{num(r.confidence_low_kw).toFixed(2)}</td>
                  <td className="mono text-slate-500">{num(r.confidence_high_kw).toFixed(2)}</td>
                  <td className="mono text-slate-600">{num(r.irradiance_proxy).toFixed(1)}</td>
                  <td className="mono text-slate-600">{num(r.cloud_pct).toFixed(1)}</td>
                  <td className="mono text-slate-600">{num(r.temp_c).toFixed(1)}</td>
                  <td><span className={`status-pill ${r.status === "active" ? "status-pill-on" : "status-pill-off"}`}>{r.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
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
    <div className={`card-panel overflow-hidden relative p-6 border-t-4 
      ${accent === 'amber' ? 'border-t-[#D97706]' : 
        accent === 'green' ? 'border-t-[#059669]' : 
        accent === 'blue' ? 'border-t-[#2563EB]' : 'border-t-[#64748B]'}`}>
      <div className="flex justify-between items-center mb-4">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-500">{title}</div>
        <span className={`status-chip ${status}`}>{status}</span>
      </div>
      <div className="text-3xl font-bold mono text-[#1E293B]">
        {valueText ?? (typeof value === "number" ? `${num(value).toFixed(2)} ${unit ?? ""}` : "--")}
      </div>
    </div>
  );
}
