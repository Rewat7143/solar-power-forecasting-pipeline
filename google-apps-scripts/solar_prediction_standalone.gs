/**
 * Standalone Solar Prediction Web App (Google Apps Script)
 *
 * This app is intentionally independent from your existing website.
 * Deploy as a Web App and open the deployment URL directly.
 */

const SOLAR_APP_CONFIG = {
  TZ: 'Asia/Kolkata',
  INTERVAL_MIN: 5,
  HISTORY_DAYS: 45,
  SOLAR_SPREADSHEET_ID: 'PUT_YOUR_SPREADSHEET_ID_HERE',
  SOLAR_SHEET_NAME: 'energy_5min',
  WEATHER_SPREADSHEET_ID: 'PUT_YOUR_SPREADSHEET_ID_HERE',
  WEATHER_SHEET_NAME: 'weather_5min',
  DEFAULT_TEMP_C: 32,
  DEFAULT_CLOUD_PCT: 35,
  DEFAULT_IRRADIANCE_WM2: 700,
  MIN_STD_KW: 2.0,
  MAX_IRRADIANCE_WM2: 1200,
};

function doGet() {
  return HtmlService
    .createHtmlOutputFromFile('solar_prediction_ui')
    .setTitle('Solar Prediction Monitor')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function getSolarPrediction(dateStr, timeStr) {
  const targetTs = parseDateTimeOrNow_(dateStr, timeStr);
  const solarRows = readSolarRows_();
  const weatherRows = readWeatherRows_();

  if (!solarRows.length) {
    throw new Error('No solar rows found. Check sheet ID/name and headers.');
  }

  const historyCutoff = new Date(targetTs.getTime() - SOLAR_APP_CONFIG.HISTORY_DAYS * 24 * 60 * 60 * 1000);
  const historyRows = solarRows.filter(r => r.timestamp >= historyCutoff && r.timestamp < targetTs);
  if (!historyRows.length) {
    throw new Error('Not enough historical rows before requested timestamp.');
  }

  const slotStats = buildSlotStats_(historyRows);
  const weatherAtTarget = findNearestWeather_(weatherRows, targetTs);
  const prediction = predictAtTimestamp_(targetTs, slotStats, weatherAtTarget);
  const context = buildContextSeries_(targetTs, slotStats, weatherRows);

  const recentRows = historyRows.slice(Math.max(0, historyRows.length - 288));
  const avgRecent = recentRows.length
    ? recentRows.reduce((s, r) => s + r.solar_power_kw, 0) / recentRows.length
    : prediction.solar_power_kw;
  const peakRecent = recentRows.length
    ? recentRows.reduce((m, r) => Math.max(m, r.solar_power_kw), 0)
    : prediction.solar_power_kw;

  return {
    ok: true,
    generatedAt: formatIso_(new Date()),
    requestDate: Utilities.formatDate(targetTs, SOLAR_APP_CONFIG.TZ, 'yyyy-MM-dd'),
    requestTime: Utilities.formatDate(targetTs, SOLAR_APP_CONFIG.TZ, 'HH:mm'),
    summary: {
      currentSolarKw: round2_(prediction.solar_power_kw),
      peakSolarKw: round2_(peakRecent),
      avgSolarKw: round2_(avgRecent),
      confidenceBandKw: {
        low: round2_(prediction.confidence_low_kw),
        high: round2_(prediction.confidence_high_kw),
      },
      status: prediction.status,
    },
    note: 'Prediction is generated for one selected timestamp, not as a 24-hour future forecast.',
    chartSeries: context,
    forecastColumns: [prediction],
  };
}

function parseDateTimeOrNow_(dateStr, timeStr) {
  const dateOk = dateStr && /^\d{4}-\d{2}-\d{2}$/.test(String(dateStr));
  const timeOk = timeStr && /^\d{2}:\d{2}$/.test(String(timeStr));

  if (dateOk && timeOk) {
    const d = new Date(String(dateStr) + 'T' + String(timeStr) + ':00');
    if (!isNaN(d.getTime())) return d;
  }
  return new Date();
}

function readSolarRows_() {
  const sheet = openSheet_(SOLAR_APP_CONFIG.SOLAR_SPREADSHEET_ID, SOLAR_APP_CONFIG.SOLAR_SHEET_NAME);
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) return [];

  const headers = values[0].map(v => String(v).trim().toLowerCase());
  const idxTs = headers.indexOf('timestamp');
  const idxSolar = headers.indexOf('solar_power_kw');
  if (idxTs < 0 || idxSolar < 0) {
    throw new Error('Solar sheet headers must include timestamp and solar_power_kw');
  }

  const out = [];
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    const ts = new Date(row[idxTs]);
    if (isNaN(ts.getTime())) continue;
    out.push({
      timestamp: ts,
      solar_power_kw: toNum_(row[idxSolar]),
    });
  }
  out.sort((a, b) => a.timestamp - b.timestamp);
  return out;
}

function readWeatherRows_() {
  const sheet = openSheet_(SOLAR_APP_CONFIG.WEATHER_SPREADSHEET_ID, SOLAR_APP_CONFIG.WEATHER_SHEET_NAME);
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) return [];

  const headers = values[0].map(v => String(v).trim().toLowerCase());
  const idxTs = headers.indexOf('timestamp');
  const idxIrr = headers.indexOf('irradiance_wm2');
  const idxCloud = headers.indexOf('cloud_pct');
  const idxTemp = headers.indexOf('temp_c');
  if (idxTs < 0 || idxIrr < 0 || idxCloud < 0 || idxTemp < 0) {
    throw new Error('Weather sheet headers must include timestamp, irradiance_wm2, cloud_pct, temp_c');
  }

  const out = [];
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    const ts = new Date(row[idxTs]);
    if (isNaN(ts.getTime())) continue;
    out.push({
      timestamp: ts,
      irradiance_wm2: toNum_(row[idxIrr]),
      cloud_pct: toNum_(row[idxCloud]),
      temp_c: toNum_(row[idxTemp]),
    });
  }
  out.sort((a, b) => a.timestamp - b.timestamp);
  return out;
}

function buildSlotStats_(rows) {
  const slots = Math.floor((24 * 60) / SOLAR_APP_CONFIG.INTERVAL_MIN);
  const bucket = Array(slots).fill(0).map(() => []);
  rows.forEach(r => {
    bucket[getSlotIndex_(r.timestamp)].push(r.solar_power_kw);
  });

  return bucket.map(arr => {
    if (!arr.length) return { mean: 0, std: SOLAR_APP_CONFIG.MIN_STD_KW };
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(arr.length - 1, 1);
    return { mean: mean, std: Math.max(Math.sqrt(variance), SOLAR_APP_CONFIG.MIN_STD_KW) };
  });
}

function findNearestWeather_(weatherRows, ts) {
  if (!weatherRows.length) {
    return {
      irradiance_wm2: SOLAR_APP_CONFIG.DEFAULT_IRRADIANCE_WM2,
      cloud_pct: SOLAR_APP_CONFIG.DEFAULT_CLOUD_PCT,
      temp_c: SOLAR_APP_CONFIG.DEFAULT_TEMP_C,
      source: 'default',
    };
  }

  let best = weatherRows[0];
  let bestDiff = Math.abs(weatherRows[0].timestamp - ts);
  for (let i = 1; i < weatherRows.length; i++) {
    const d = Math.abs(weatherRows[i].timestamp - ts);
    if (d < bestDiff) {
      best = weatherRows[i];
      bestDiff = d;
    }
  }
  return {
    irradiance_wm2: best.irradiance_wm2,
    cloud_pct: best.cloud_pct,
    temp_c: best.temp_c,
    source: 'sheet_nearest',
  };
}

function predictAtTimestamp_(ts, slotStats, weather) {
  const slot = getSlotIndex_(ts);
  const base = slotStats[slot] || { mean: 0, std: SOLAR_APP_CONFIG.MIN_STD_KW };

  const irr = clamp_(weather.irradiance_wm2, 0, SOLAR_APP_CONFIG.MAX_IRRADIANCE_WM2);
  const cloud = clamp_(weather.cloud_pct, 0, 100);
  const temp = toNum_(weather.temp_c);

  const irrFactor = 0.55 + 0.75 * (irr / 1000);
  const cloudFactor = 1 - 0.45 * (cloud / 100);
  const tempFactor = 1 - Math.min(0.18, Math.abs(temp - 32) * 0.005);

  const predRaw = base.mean * irrFactor * cloudFactor * tempFactor;
  const pred = Math.max(0, predRaw);

  const sigma = Math.max(base.std, SOLAR_APP_CONFIG.MIN_STD_KW);
  const low = Math.max(0, pred - 1.28 * sigma);
  const high = pred + 1.28 * sigma;

  return {
    timestamp: formatIso_(ts),
    timeLabel: Utilities.formatDate(ts, SOLAR_APP_CONFIG.TZ, 'HH:mm'),
    solar_power_kw: round2_(pred),
    confidence_low_kw: round2_(low),
    confidence_high_kw: round2_(high),
    irradiance_proxy: round2_(irr),
    cloud_pct: round2_(cloud),
    temp_c: round2_(temp),
    status: pred > 1 ? 'active' : 'standby',
    source: weather.source,
  };
}

function buildContextSeries_(targetTs, slotStats, weatherRows) {
  const labels = [];
  const solarPowerKw = [];
  const lowBandKw = [];
  const highBandKw = [];
  const meta = [];

  const dayStart = new Date(targetTs);
  dayStart.setHours(0, 0, 0, 0);
  const steps = Math.floor((24 * 60) / SOLAR_APP_CONFIG.INTERVAL_MIN);
  let selectedIdx = 0;

  for (let i = 0; i < steps; i++) {
    const ts = new Date(dayStart.getTime() + i * SOLAR_APP_CONFIG.INTERVAL_MIN * 60 * 1000);
    const slot = getSlotIndex_(ts);
    const base = slotStats[slot] || { mean: 0, std: SOLAR_APP_CONFIG.MIN_STD_KW };
    const weather = findNearestWeather_(weatherRows, ts);
    const pred = predictAtTimestamp_(ts, slotStats, weather);

    labels.push(pred.timeLabel);
    solarPowerKw.push(pred.solar_power_kw);
    lowBandKw.push(pred.confidence_low_kw);
    highBandKw.push(pred.confidence_high_kw);
    meta.push({
      timestamp: pred.timestamp,
      irradiance_wm2: pred.irradiance_proxy,
      cloud_pct: pred.cloud_pct,
      temp_c: pred.temp_c,
      status: pred.status,
      slot_mean_kw: round2_(base.mean),
      slot_std_kw: round2_(base.std),
    });

    if (Math.abs(ts - targetTs) <= SOLAR_APP_CONFIG.INTERVAL_MIN * 60 * 1000) {
      selectedIdx = i;
    }
  }

  return {
    labels: labels,
    solarPowerKw: solarPowerKw,
    lowBandKw: lowBandKw,
    highBandKw: highBandKw,
    selectedIdx: selectedIdx,
    meta: meta,
  };
}

function openSheet_(spreadsheetId, sheetName) {
  if (!spreadsheetId || spreadsheetId === 'PUT_YOUR_SPREADSHEET_ID_HERE') {
    throw new Error('Set spreadsheet IDs in SOLAR_APP_CONFIG first.');
  }
  const ss = SpreadsheetApp.openById(spreadsheetId);
  const sheet = ss.getSheetByName(sheetName);
  if (!sheet) throw new Error('Sheet not found: ' + sheetName);
  return sheet;
}

function getSlotIndex_(ts) {
  const mins = ts.getHours() * 60 + ts.getMinutes();
  return Math.floor(mins / SOLAR_APP_CONFIG.INTERVAL_MIN);
}

function formatIso_(d) {
  return Utilities.formatDate(d, SOLAR_APP_CONFIG.TZ, "yyyy-MM-dd'T'HH:mm:ssXXX");
}

function round2_(v) {
  return Math.round(v * 100) / 100;
}

function toNum_(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function clamp_(v, low, high) {
  return Math.max(low, Math.min(high, v));
}
