"""
pages/2_Vessel_Map.py — Interactive Folium map of the port geofence zones.
"""

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import ANCHORAGE_ZONE, BERTH_ZONE

st.set_page_config(page_title="Vessel Map · Port Congestion", page_icon="🗺️", layout="wide")

st.title("🗺️ Port Zone Map")
st.caption("San Pedro Bay — Anchorage & Berth Geofence Zones")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col2:
    show_anchorage = st.checkbox("Show Anchorage Zone", value=True)
    show_berth = st.checkbox("Show Berth Zone", value=True)
    map_style = st.selectbox(
        "Map style",
        ["CartoDB dark_matter", "OpenStreetMap", "CartoDB positron"],
    )

# ── Build Folium map ──────────────────────────────────────────────────────────
CENTER = [33.67, -118.22]
m = folium.Map(location=CENTER, zoom_start=11, tiles=map_style)

if show_anchorage:
    coords = [(lat, lon) for lon, lat in ANCHORAGE_ZONE.exterior.coords]
    folium.Polygon(
        locations=coords,
        color="#FF6600",
        fill=True,
        fill_color="#FF6600",
        fill_opacity=0.12,
        weight=2,
        tooltip="Anchorage Zone — vessels queue here while awaiting a berth",
        popup=folium.Popup(
            "<b>Anchorage Zone</b><br>"
            "Vessels anchor here while waiting for a berth to become available.<br>"
            "During Oct 2021, this zone held 100+ vessels simultaneously.",
            max_width=280,
        ),
    ).add_to(m)

    folium.Marker(
        location=[33.63, -118.22],
        tooltip="Anchorage Zone",
        icon=folium.DivIcon(html=(
            '<div style="color:#FF6600;font-weight:bold;font-size:13px;'
            'background:rgba(0,0,0,0.6);padding:3px 6px;border-radius:4px;">'
            "⚓ Anchorage</div>"
        )),
    ).add_to(m)

if show_berth:
    coords = [(lat, lon) for lon, lat in BERTH_ZONE.exterior.coords]
    folium.Polygon(
        locations=coords,
        color="#0066CC",
        fill=True,
        fill_color="#0066CC",
        fill_opacity=0.18,
        weight=2,
        tooltip="Berth Zone — active terminal / port area",
        popup=folium.Popup(
            "<b>Berth / Terminal Zone</b><br>"
            "Inside the San Pedro Bay breakwater.<br>"
            "Port of LA + Port of Long Beach operate ~27 container berths in this area.",
            max_width=280,
        ),
    ).add_to(m)

    folium.Marker(
        location=[33.74, -118.22],
        tooltip="Berth Zone",
        icon=folium.DivIcon(html=(
            '<div style="color:#0099FF;font-weight:bold;font-size:13px;'
            'background:rgba(0,0,0,0.6);padding:3px 6px;border-radius:4px;">'
            "🏗️ Berth Zone</div>"
        )),
    ).add_to(m)

# Port landmarks
landmarks = [
    ([33.7453, -118.2683], "Port of Los Angeles Main Gate", "🏭"),
    ([33.7749, -118.2146], "Port of Long Beach Terminal", "🏭"),
    ([33.7083, -118.2922], "San Pedro Bay Breakwater", "🌊"),
]
for loc, name, icon in landmarks:
    folium.Marker(
        location=loc,
        tooltip=name,
        icon=folium.DivIcon(html=(
            f'<div style="font-size:18px" title="{name}">{icon}</div>'
        )),
    ).add_to(m)

# ── Render map ────────────────────────────────────────────────────────────────
with col1:
    st_folium(m, width=None, height=560, returned_objects=[])

# ── Info cards ────────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    **⚓ Anchorage Zone**
    - Approximate bounds: 33.55°N–33.75°N, 118.05°W–118.35°W
    - Vessels are classified as *waiting* when detected in this zone
    - Wait time starts when a vessel enters here, ends when it moves to the berth zone
    """)

with c2:
    st.markdown("""
    **🏗️ Berth Zone**
    - Inside the San Pedro Bay breakwater
    - ~27 active container berths across Port of LA and Port of Long Beach
    - Mean service time: ~48 hours per vessel (used in M/M/c baseline)
    """)

with c3:
    st.markdown("""
    **📡 AIS Classification**
    - Every vessel ping is classified into `anchorage / berth / transit`
    - Zone changes are detected as events (arrival, berthing, departure)
    - Only cargo (type 70–79) and tanker (type 80–89) vessels are included
    """)

st.divider()
st.info(
    "**Live AIS integration:** For real-time vessel positions, connect an AISHub "
    "or MarineTraffic API key and update `src/pipeline.py` to poll the live feed. "
    "The geofence classification logic is already in place.",
    icon="ℹ️",
)
