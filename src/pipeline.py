"""
pipeline.py — AIS data download, zone classification, event detection,
and congestion index construction for Port of LA / Long Beach.
"""

import io
import time
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon

# ── Geofence config ─────────────────────────────────────────────────────────
# San Pedro Bay — anchorage area where vessels queue while waiting for a berth
ANCHORAGE_ZONE = Polygon([
    (-118.35, 33.55), (-118.05, 33.55),
    (-118.05, 33.75), (-118.35, 33.75),
])

# Inside the breakwater — actual terminal / berth area
BERTH_ZONE = Polygon([
    (-118.27, 33.70), (-118.17, 33.70),
    (-118.17, 33.78), (-118.27, 33.78),
])

# AIS VesselType codes for cargo (70-79) and tanker (80-89) vessels only
CARGO_TANKER_TYPES = set(range(70, 90))

# Minimum healthy row count per day — days below this are treated as partial
HEALTHY_ROW_THRESHOLD = 30_000

LAT_RANGE = (33.5, 33.8)
LON_RANGE = (-118.4, -118.0)


def classify_zone(lon: float, lat: float) -> str:
    pt = Point(lon, lat)
    if BERTH_ZONE.contains(pt):
        return "berth"
    if ANCHORAGE_ZONE.contains(pt):
        return "anchorage"
    return "transit"


def filter_to_port_area(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["LAT"].between(*LAT_RANGE)) & (df["LON"].between(*LON_RANGE))].copy()
    if "VesselType" in df.columns:
        df = df[df["VesselType"].isin(CARGO_TANKER_TYPES)]
    return df


def download_ais_day(
    year: int, month: int, day: int,
    max_retries: int = 3,
    retry_delay: int = 10,
    verbose: bool = True,
) -> pd.DataFrame | None:
    url = (
        f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
        f"{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(csv_name))
            if len(df) < HEALTHY_ROW_THRESHOLD:
                raise ValueError(f"Partial download: only {len(df)} rows")
            if verbose:
                print(f"  ✅ {year}-{month:02d}-{day:02d}: {len(df):,} rows")
            return df
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    if verbose:
        print(f"  ❌ {year}-{month:02d}-{day:02d}: permanently failed — skipping")
    return None


def download_date_range(
    start_date: datetime,
    n_days: int,
    verbose: bool = True,
) -> pd.DataFrame:
    frames = []
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        if verbose:
            print(f"Downloading {d.date()}...")
        df = download_ais_day(d.year, d.month, d.day, verbose=verbose)
        if df is not None:
            df = filter_to_port_area(df)
            frames.append(df)

    if not frames:
        raise RuntimeError("No AIS data could be downloaded.")

    ais = pd.concat(frames, ignore_index=True)
    ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"])
    ais = ais.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
    if verbose:
        print(f"\nTotal: {len(ais):,} records | {ais['MMSI'].nunique():,} unique vessels")
    return ais


def extract_events(ais_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse AIS pings to zone-change events per vessel."""
    records = []
    for mmsi, group in ais_df.groupby("MMSI"):
        prev_zone = None
        for _, row in group.iterrows():
            z = classify_zone(row["LON"], row["LAT"])
            if z != prev_zone:
                records.append({
                    "MMSI": mmsi,
                    "timestamp": row["BaseDateTime"],
                    "zone": z,
                    "VesselType": row.get("VesselType"),
                })
                prev_zone = z
    return pd.DataFrame(records)


def compute_wait_times(event_log: pd.DataFrame) -> pd.DataFrame:
    """Return realized anchorage→berth wait time per vessel call."""
    rows = []
    for mmsi, g in event_log.groupby("MMSI"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "zone"] == "anchorage" and g.loc[i + 1, "zone"] == "berth":
                wait_hr = (
                    g.loc[i + 1, "timestamp"] - g.loc[i, "timestamp"]
                ).total_seconds() / 3600
                rows.append({
                    "MMSI": mmsi,
                    "arrival": g.loc[i, "timestamp"],
                    "berthed": g.loc[i + 1, "timestamp"],
                    "wait_hours": wait_hr,
                    "VesselType": g.loc[i, "VesselType"],
                })
    return pd.DataFrame(rows)


def build_congestion_features(ais_df: pd.DataFrame) -> pd.DataFrame:
    """Build hourly congestion index from raw AIS dataframe."""
    ais_df = ais_df.copy()
    ais_df["zone"] = ais_df.apply(
        lambda r: classify_zone(r["LON"], r["LAT"]), axis=1
    )
    ais_df["hour"] = ais_df["BaseDateTime"].dt.floor("h")

    full_hours = pd.date_range(
        ais_df["hour"].min(), ais_df["hour"].max(), freq="h"
    )

    # Queue length: distinct vessels in anchorage each hour
    queue_length = (
        ais_df[ais_df["zone"] == "anchorage"]
        .groupby("hour")["MMSI"]
        .nunique()
        .reindex(full_hours, fill_value=0)
        .rename("queue_length")
    )

    # Total pings any zone: low pings = data gap
    total_pings = (
        ais_df.groupby("hour").size()
        .reindex(full_hours, fill_value=0)
        .rename("total_pings")
    )

    # Arrival rate: new anchorage entries per hour
    event_log = extract_events(ais_df)
    arrivals = event_log[event_log["zone"] == "anchorage"].copy()
    arrivals["hour"] = arrivals["timestamp"].dt.floor("h")
    arrival_rate = (
        arrivals.groupby("hour").size()
        .reindex(full_hours, fill_value=0)
        .rename("arrival_rate")
    )

    df = pd.DataFrame({
        "queue_length": queue_length,
        "arrival_rate": arrival_rate,
        "total_pings": total_pings,
    }, index=full_hours).reset_index().rename(columns={"index": "hour"})

    df["data_gap"] = df["total_pings"] < 5
    df["dayofweek"] = df["hour"].dt.dayofweek
    df["hourofday"] = df["hour"].dt.hour

    return df
