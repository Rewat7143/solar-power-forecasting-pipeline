/**
 * Solar Forecast WebApp (Google Apps Script)
 *
 * Purpose:
 * - Provide 24h / 5-min solar forecast data for a dashboard page.
 * - Output format is aligned with card + chart + table columns like your sample UI.
 *
 * Expected Sheet columns (header row):
 * - timestamp
 * - solar_power_kw
 * - grid_supply_kw
 * - dg_sets_kw
 *
 * Deploy:
 * 1) Put this file in your Apps Script project.
 * 2) Set CONFIG.SPREADSHEET_ID and CONFIG.SHEET_NAME.
 * 3) Deploy as Web App (Execute as Me, Anyone with link).
 * 4) Call: https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec?date=2026-04-18
 */

const CONFIG = {
  SPREADSHEET_ID: 'PUT_YOUR_SPREADSHEET_ID_HERE',
  SHEET_NAME: 'energy_5min',
  TIMEZONE: 'Asia/Kolkata',
  INTERVAL_MINUTES: 5,
  HORIZON_HOURS: 24,
  HISTORY_DAYS: 30,
  MIN_SOLAR_KW: 0,
  MIN_GRID_KW: 0,
  MIN_DG_KW: 0,
  DEFAULT_STD_DEV: 2.5,
};

function doGet(e) {
  try {
    const dateParam = (e && e.parameter && e.parameter.date) ? String(e.parameter.date) : '';
    const requestDate = parseRequestDateOrToday(dateParam);

    const payload = buildForecastPayload(requestDate);
    return ContentService
      .createTextOutput(JSON.stringify(payload, null, 2))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({
        ok: false,
        error: String(err),
      }, null, 2))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function parseRequestDateOrToday(dateStr) {
  if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    const d = new Date(dateStr + 'T00:00:00');
    if (!isNaN(d.getTime())) return d;
  }
  const now = new Date();
  const todayStr = Utilities.formatDate(now, CONFIG.TIMEZONE, 'yyyy-MM-dd');
  return new Date(todayStr + 'T00:00:00');
}

function buildForecastPayload(baseDate) {
  const rows = readEnergyRows();
  if (!rows.length) {
    throw new Error('No rows found in source sheet.');
  }

  const historyCutoff = new Date(baseDate.getTime() - CONFIG.HISTORY_DAYS * 24 * 60 * 60 * 1000);
  const historyRows = rows.filter(r => r.timestamp >= historyCutoff && r.timestamp < baseDate);
  if (!historyRows.length) {
    throw new Error('Not enough historical rows for forecasting.');
  }

  const slotStats = buildSlotStats(historyRows);
  const startTs = new Date(baseDate);
  const steps = Math.floor((CONFIG.HORIZON_HOURS * 60) / CONFIG.INTERVAL_MINUTES);

  const forecastColumns = [];
  const labels = [];
  const solarSeries = [];
  const gridSeries = [];
  const dgSeries = [];
  const totalSeries = [];

  for (let i = 0; i < steps; i++) {
    const ts = new Date(startTs.getTime() + i * CONFIG.INTERVAL_MINUTES * 60 * 1000);
    const slot = getSlotIndex(ts);
    const hour = ts.getHours() + ts.getMinutes() / 60;

    const solarBase = getStat(slotStats.solar, slot, 'mean', 0);
    const gridBase = getStat(slotStats.grid, slot, 'mean', 0);
    const dgBase = getStat(slotStats.dg, slot, 'mean', 0);

    const dayFactor = solarDayShapeFactor(hour);
    const solarKw = clamp(solarBase * dayFactor, CONFIG.MIN_SOLAR_KW);

    const gridKw = clamp(gridBase, CONFIG.MIN_GRID_KW);
    const dgKw = clamp(dgBase, CONFIG.MIN_DG_KW);
    const totalKw = clamp(solarKw + gridKw + dgKw, 0);

    const solarPct = totalKw > 0 ? (solarKw / totalKw) * 100 : 0;
    const confidence = buildConfidenceBand(slotStats.solar, slot, solarKw);

    const row = {
      timestamp: Utilities.formatDate(ts, CONFIG.TIMEZONE, "yyyy-MM-dd'T'HH:mm:ssXXX"),
      timeLabel: Utilities.formatDate(ts, CONFIG.TIMEZONE, 'HH:mm'),
      solar_power_kw: round2(solarKw),
      grid_supply_kw: round2(gridKw),
      dg_sets_kw: round2(dgKw),
      total_supply_kw: round2(totalKw),
      solar_share_pct: round2(solarPct),
      status: inferStatus(round2(gridKw), round2(solarKw), round2(dgKw)),
      confidence_low_kw: round2(confidence.low),
      confidence_high_kw: round2(confidence.high),
    };

    forecastColumns.push(row);
    labels.push(row.timeLabel);
    solarSeries.push(row.solar_power_kw);
    gridSeries.push(row.grid_supply_kw);
    dgSeries.push(row.dg_sets_kw);
    totalSeries.push(row.total_supply_kw);
  }

  const card = buildCardSummary(forecastColumns);

  return {
    ok: true,
    generatedAt: Utilities.formatDate(new Date(), CONFIG.TIMEZONE, "yyyy-MM-dd'T'HH:mm:ssXXX"),
    request: {
      date: Utilities.formatDate(baseDate, CONFIG.TIMEZONE, 'yyyy-MM-dd'),
      intervalMinutes: CONFIG.INTERVAL_MINUTES,
      horizonHours: CONFIG.HORIZON_HOURS,
    },
    cards: card,
    forecastColumns: forecastColumns,
    chartSeries: {
      labels: labels,
      solarPowerKw: solarSeries,
      gridSupplyKw: gridSeries,
      dgSetsKw: dgSeries,
      totalSupplyKw: totalSeries,
    },
  };
}

function readEnergyRows() {
  if (!CONFIG.SPREADSHEET_ID || CONFIG.SPREADSHEET_ID === 'PUT_YOUR_SPREADSHEET_ID_HERE') {
    throw new Error('Set CONFIG.SPREADSHEET_ID before running.');
  }

  const ss = SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
  const sheet = ss.getSheetByName(CONFIG.SHEET_NAME);
  if (!sheet) {
    throw new Error('Sheet not found: ' + CONFIG.SHEET_NAME);
  }

  const values = sheet.getDataRange().getValues();
  if (values.length < 2) return [];

  const headers = values[0].map(v => String(v).trim().toLowerCase());
  const idx = {
    timestamp: headers.indexOf('timestamp'),
    solar: headers.indexOf('solar_power_kw'),
    grid: headers.indexOf('grid_supply_kw'),
    dg: headers.indexOf('dg_sets_kw'),
  };

  if (idx.timestamp < 0 || idx.solar < 0 || idx.grid < 0 || idx.dg < 0) {
    throw new Error('Missing required headers. Need: timestamp, solar_power_kw, grid_supply_kw, dg_sets_kw');
  }

  const out = [];
  for (let r = 1; r < values.length; r++) {
    const row = values[r];
    const ts = new Date(row[idx.timestamp]);
    if (isNaN(ts.getTime())) continue;

    out.push({
      timestamp: ts,
      solar_power_kw: toNum(row[idx.solar]),
      grid_supply_kw: toNum(row[idx.grid]),
      dg_sets_kw: toNum(row[idx.dg]),
    });
  }

  out.sort((a, b) => a.timestamp - b.timestamp);
  return out;
}

function buildSlotStats(rows) {
  const slotsPerDay = Math.floor((24 * 60) / CONFIG.INTERVAL_MINUTES);
  const slotMap = {
    solar: Array(slotsPerDay).fill(0).map(() => []),
    grid: Array(slotsPerDay).fill(0).map(() => []),
    dg: Array(slotsPerDay).fill(0).map(() => []),
  };

  rows.forEach(r => {
    const s = getSlotIndex(r.timestamp);
    slotMap.solar[s].push(r.solar_power_kw);
    slotMap.grid[s].push(r.grid_supply_kw);
    slotMap.dg[s].push(r.dg_sets_kw);
  });

  return {
    solar: slotMap.solar.map(statForArray),
    grid: slotMap.grid.map(statForArray),
    dg: slotMap.dg.map(statForArray),
  };
}

function statForArray(arr) {
  if (!arr || !arr.length) {
    return { mean: 0, std: CONFIG.DEFAULT_STD_DEV };
  }
  const n = arr.length;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(n - 1, 1);
  return {
    mean: mean,
    std: Math.sqrt(Math.max(variance, 0)),
  };
}

function solarDayShapeFactor(hour) {
  if (hour < 5 || hour > 19.5) return 0;

  // Bell-like day curve centered near noon (simple physical prior).
  const x = (hour - 12) / 4.5;
  const factor = Math.exp(-0.5 * x * x);

  // Keep small floor during daylight to avoid hard discontinuities.
  return Math.max(0.12, Math.min(1.0, factor));
}

function buildConfidenceBand(statArray, slot, value) {
  const std = getStat(statArray, slot, 'std', CONFIG.DEFAULT_STD_DEV);
  return {
    low: clamp(value - 1.28 * std, 0),
    high: clamp(value + 1.28 * std, 0),
  };
}

function buildCardSummary(forecastRows) {
  if (!forecastRows.length) {
    return {
      grid_supply_kw: 0,
      solar_power_kw: 0,
      dg_sets_kw: 0,
      total_supply_kw: 0,
      solar_share_pct: 0,
      status: 'standby',
    };
  }

  const first = forecastRows[0];
  return {
    grid_supply_kw: first.grid_supply_kw,
    solar_power_kw: first.solar_power_kw,
    dg_sets_kw: first.dg_sets_kw,
    total_supply_kw: first.total_supply_kw,
    solar_share_pct: first.solar_share_pct,
    status: first.status,
  };
}

function inferStatus(gridKw, solarKw, dgKw) {
  if (dgKw > 0.1) return 'standby';
  if (solarKw > gridKw) return 'active';
  if (gridKw > 0.1) return 'import';
  return 'live';
}

function getSlotIndex(ts) {
  const mins = ts.getHours() * 60 + ts.getMinutes();
  return Math.floor(mins / CONFIG.INTERVAL_MINUTES);
}

function getStat(statArray, slot, key, fallback) {
  if (!statArray || !statArray[slot]) return fallback;
  const v = statArray[slot][key];
  return Number.isFinite(v) ? v : fallback;
}

function toNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function clamp(v, minVal) {
  return Math.max(minVal, v);
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
