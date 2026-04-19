# Solar Prediction Vercel App

Standalone timestamp-based solar prediction app designed for Vercel deployment.

## What This Includes

- Next.js frontend with dashboard-style cards and interactive chart
- `/api/predict` backend route for timestamp prediction
- CSV-based data hooks for solar + weather sources
- Tooltip details on hover (irradiance, cloud, temp, slot stats)

## Required CSV Headers

Solar CSV (`SOLAR_CSV_URL`):

- `timestamp`
- `solar_power_kw`

Weather CSV (`WEATHER_CSV_URL`):

- `timestamp`
- `irradiance_wm2`
- `cloud_pct`
- `temp_c`

## Local Run

```bash
npm install
cp .env.example .env.local
# fill SOLAR_CSV_URL and WEATHER_CSV_URL
npm run dev
```

Open `http://localhost:3000`

## Deploy To Vercel

1. Push this folder to a Git repo.
2. Import project in Vercel.
3. Set environment variables in Vercel:
   - `SOLAR_CSV_URL`
   - `WEATHER_CSV_URL`
4. Deploy.

## API Usage

`POST /api/predict`

Request body:

```json
{
  "date": "2026-04-18",
  "time": "13:20"
}
```

Response includes:

- `summary` for cards
- `chartSeries` for graph
- `forecastColumns` for selected timestamp row
