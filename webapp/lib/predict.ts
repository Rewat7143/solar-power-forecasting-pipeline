import { spawnSync } from "node:child_process";
import path from "node:path";

export type PredictionPoint = {
  timestamp: string;
  timeLabel: string;
  solar_power_kw: number;
  confidence_low_kw: number;
  confidence_high_kw: number;
  irradiance_proxy: number;
  cloud_pct: number;
  temp_c: number;
  status: "active" | "standby";
  source: string;
};

export type PredictionResponse = {
  ok: boolean;
  generatedAt: string;
  requestDate: string;
  requestTime: string;
  note: string;
  resolvedModel?: string;
  dataSource?: string;
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
  forecastColumns: PredictionPoint[];
};

const CACHE_TTL_MS = 5 * 60 * 1000;
const PREDICTION_CACHE_VERSION = "v2";

type CacheEntry = {
  value: PredictionResponse;
  ts: number;
};

const predictionCache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<PredictionResponse>>();

export async function runTimestampPrediction(dateStr: string, timeStr: string, requestedModel?: string): Promise<PredictionResponse> {
  const projectRoot = process.cwd();
  const repoRoot = path.resolve(projectRoot, "..");
  const scriptPath = path.join(projectRoot, "scripts", "predict_with_model.py");
  const manifestPath = path.join(repoRoot, "models", "manifest.json");
  const pythonBin = process.env.PYTHON_BIN || path.join(repoRoot, ".venv", "bin", "python");
  const dataRoot = process.env.SOLAR_DATA_ROOT || repoRoot;
  const modelName = requestedModel || process.env.MODEL_NAME || "";
  const cacheKey = `${PREDICTION_CACHE_VERSION}|${dateStr}|${timeStr}|${dataRoot}|${modelName}`;

  const cached = predictionCache.get(cacheKey);
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
    return cached.value;
  }

  const running = inFlight.get(cacheKey);
  if (running) {
    return running;
  }

  const payload = JSON.stringify({
    date: dateStr,
    time: timeStr,
    manifestPath,
    dataRoot,
    modelName,
  });

  const task = Promise.resolve()
    .then(() => {
      const proc = spawnSync(pythonBin, [scriptPath], {
        input: payload,
        encoding: "utf-8",
        timeout: 120_000,
        maxBuffer: 10 * 1024 * 1024,
      });

      if (proc.status === 0 && proc.stdout?.trim()) {
        const parsed = JSON.parse(proc.stdout) as PredictionResponse;
        predictionCache.set(cacheKey, { value: parsed, ts: Date.now() });
        return parsed;
      }

      const stderr = (proc.stderr || proc.stdout || "Prediction bridge failed.").toString().trim();
      throw new Error(stderr || "Prediction bridge failed.");
    })
    .finally(() => {
      inFlight.delete(cacheKey);
    });

  inFlight.set(cacheKey, task);
  return task;
}
