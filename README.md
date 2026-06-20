# ⚓ Port Congestion Forecasting System

AIS-based anchorage queue forecasting for the **Port of Los Angeles / Long Beach**, built around the October–November 2021 congestion crisis. Predicts vessel queue length 6 hours ahead using XGBoost, benchmarked against an M/M/c queueing-theory baseline, with SHAP explainability.

Trained entirely on **free, public, real-world data — no API key required.**

**[🚀 Live demo →](#deploy-to-streamlit-community-cloud)** *(add your deployed URL here after Step 4)*

---

## What this does

1. **Ingests** NOAA Marine Cadastre AIS data (free, historical, bulk-downloadable vessel position broadcasts — no registration or key needed)
2. **Geofences** San Pedro Bay into anchorage and berth zones, classifies every AIS ping
3. **Detects events** — vessel arrival → berthing → departure — to compute realized wait times
4. **Builds an hourly congestion index** (queue length, arrival rate, with explicit data-gap flagging)
5. **Forecasts** 6 hours ahead with XGBoost, benchmarked against an M/M/c queueing-theory baseline
6. **Explains** every prediction with SHAP (global importance + per-prediction waterfall)
7. **Serves** it all through a multi-page Streamlit dashboard with a live Folium vessel map

---

## Data source — no API key required

This project uses **[NOAA Marine Cadastre AIS data](https://hub.marinecadastre.gov/pages/vesseltraffic)**, published as free daily ZIP archives of AIS (Automatic Identification System) vessel broadcasts covering all US coastal waters. No account, registration, or API key is needed — `scripts/train.py` downloads directly over HTTPS.

If you later extend this project with **live, real-time** vessel positions (rather than this historical backtest), here's what that would require:

| Source | Free tier | API key needed? |
|---|---|---|
| **NOAA Marine Cadastre** (used here) | Full historical archive | No |
| **AISHub** | Yes, with reciprocal data sharing | Yes — free registration |
| **MarineTraffic** | Very limited | Yes — paid tiers for meaningful volume |
| **NOAA NDBC** (weather buoys, for a weather-feature extension) | Full | No |

---

## Project structure

```
port_congestion/
├── app.py                      # Streamlit home page
├── pages/
│   ├── 1_Dashboard.py          # Historical chart, backtest, live forecast widget
│   ├── 2_Vessel_Map.py         # Folium map of anchorage/berth geofence zones
│   └── 3_Explainability.py     # SHAP summary + waterfall plots
├── src/
│   ├── pipeline.py             # AIS download, geofencing, event detection
│   ├── features.py             # Feature engineering (lags, rolling stats, cyclical time)
│   ├── baseline.py             # M/M/c queueing-theory baseline
│   └── predict.py              # Model loading + inference helpers
├── scripts/
│   ├── train.py                 # ✅ Production pipeline — real NOAA data → trained model
│   └── generate_demo.py         # ⚠️ Dev-only — synthetic data, for fast local UI testing
├── models/                      # Trained model artifacts (generated, then committed)
├── data/                        # Processed features + backtest results (generated, then committed)
├── requirements.txt
└── .streamlit/config.toml       # Theme config
```

> **Note:** `models/` and `data/` ship empty in this repo. You generate real artifacts by running `scripts/train.py` yourself (see below) — nothing fake is baked in.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/port-congestion-forecasting.git
cd port-congestion-forecasting
pip install -r requirements.txt

# Downloads real NOAA AIS data (~45 days, Oct 1 – Nov 14 2021) and trains the model.
# No API key needed. Takes ~30–60 min depending on connection speed.
python scripts/train.py

streamlit run app.py
```

`scripts/train.py` has automatic retry logic and partial-download detection built in (any day under 30,000 rows is re-fetched up to 3 times before being flagged and excluded) — this came out of debugging real gaps in the NOAA archive during development, where 2 of 14 test days initially failed to download cleanly.

Swap in a different port or date range by editing `START_DATE`, `N_DAYS`, and the `ANCHORAGE_ZONE` / `BERTH_ZONE` polygons in `src/pipeline.py`.

### Just want to click through the UI without a 30-60 min wait first?

```bash
python scripts/generate_demo.py
```

This is a **dev-only convenience tool** — synthetic data, fake model, clearly tagged `synthetic` in `models/metrics.json`. The app displays a persistent warning banner whenever a synthetic model is loaded, so it's impossible to mistake for real results. **Do not commit its output as your production deployment.**

---

## Deploy to Streamlit Community Cloud

**1. Train on real data:**
```bash
python scripts/train.py
```

**2. Push to GitHub:**
```bash
git init
git add .
git commit -m "Port congestion forecasting system"
git branch -M main
git remote add origin https://github.com/<your-username>/port-congestion-forecasting.git
git push -u origin main
```

> ⚠️ The `.gitignore` excludes raw AIS downloads (`data/raw/`, `*.zip`) since they're too large for GitHub, but **explicitly keeps** the processed `data/*.csv` and `models/*` files needed for the app to run. Double-check `git status` shows your `models/` and `data/` files staged before committing.

**3. Deploy:**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click **New app**, connect your GitHub repo
- Main file path: `app.py`
- Click **Deploy**

**4. Done.** Streamlit Cloud installs `requirements.txt` and serves the app — no server config needed.

---

## Model performance

Run `scripts/train.py` to populate `models/metrics.json` with real numbers. On the real Oct–Nov 2021 NOAA AIS data, expect XGBoost to beat the M/M/c baseline meaningfully — the lag and rolling-window features give it information about queue momentum that a steady-state queueing formula structurally can't capture. Exact figures depend on which days successfully download; share your `metrics.json` output if you want a sanity check on the numbers.

---

## Why LA/Long Beach + October 2021

The Oct–Nov 2021 congestion crisis pushed anchorage queues past 100 vessels and average wait times from under a day to over two weeks — a level of dynamic range that gives a forecasting model an actual signal to learn from, rather than noise around a flat baseline. It also has a well-documented public narrative, which is useful if you're presenting this as a portfolio project.

---

## Extending this project

- **Live AIS feed** — swap the NOAA bulk download in `src/pipeline.py` for a live AISHub poll (free registration required) or MarineTraffic API (paid for meaningful volume)
- **Weather features** — pull NOAA NDBC marine buoy data (free, no key) to add storm/swell severity as a predictor
- **Multi-port** — parameterize `ANCHORAGE_ZONE` / `BERTH_ZONE` per port and add a port selector to the dashboard
- **Longer horizons** — train separate models for 12h/24h/48h forecasts alongside the current 6h model
