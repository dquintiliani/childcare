# app.py

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

from ward import WARD_REGION_MAP as wd

# -------------------------
# CONFIG
# -------------------------

load_dotenv(".env.local")  # ok even if empty / unused

BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
DATASET_ID = "licensed-child-care-centres"
RAW_CSV_PATH = "licensed_child_care_centres.csv"
CLEAN_CSV_PATH = "childcare.csv"
REQUEST_TIMEOUT = 20  # seconds
MAX_DATA_AGE_DAYS = 7  # how old the local CSV is allowed to be


# -------------------------
# DATA PIPELINE (ETL)
# -------------------------

def fetch_datastore_resource_id() -> Optional[str]:
    """Fetch the active datastore resource id for the dataset."""
    url = f"{BASE_URL}/api/3/action/package_show"
    params = {"id": DATASET_ID}

    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    package = resp.json()

    for resource in package["result"]["resources"]:
        if resource.get("datastore_active"):
            return resource["id"]

    return None


def download_raw_csv(resource_id: str, output_path: str = RAW_CSV_PATH) -> None:
    """Download the raw CSV dump from CKAN."""
    dump_url = f"{BASE_URL}/datastore/dump/{resource_id}"
    resp = requests.get(dump_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(resp.text)


def clean_and_save_data(
    raw_path: str = RAW_CSV_PATH,
    output_path: str = CLEAN_CSV_PATH,
) -> None:
    """Clean the raw CSV and write a Streamlit-ready CSV."""
    df = pd.read_csv(raw_path)

    rename_map = {
        "_id": "record_id",
        "LOC_ID": "location_id",
        "LOC_NAME": "location_name",
        "AUSPICE": "organization_type",
        "ADDRESS": "address",
        "PCODE": "postal_code",
        "ward": "city_ward",
        "PHONE": "phone",
        "bldg_type": "building_type",
        "BLDGNAME": "building_name",
        "IGSPACE": "infant_spaces",
        "TGSPACE": "toddler_spaces",
        "PGSPACE": "preschool_spaces",
        "KGSPACE": "kindergarten_spaces",
        "SGSPACE": "school_age_spaces",
        "TOTSPACE": "total_spaces",
        "subsidy": "fee_subsidy_available",
        "run_date": "data_run_date",
        "cwelcc_flag": "cwelcc_program",
        "geometry": "geo_point",
    }

    df = df.rename(columns=rename_map)
    df["region"] = df["city_ward"].map(wd)

    df.to_csv(output_path, index=False)


def is_data_stale(path: str, max_age_days: int = MAX_DATA_AGE_DAYS) -> bool:
    """Check whether a file is older than the configured max age."""
    if not os.path.exists(path):
        return True

    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime > timedelta(days=max_age_days)


def refresh_data() -> None:
    """End-to-end refresh of the childcare dataset."""
    resource_id = fetch_datastore_resource_id()
    if not resource_id:
        raise RuntimeError("No active datastore resource found for dataset.")

    download_raw_csv(resource_id, RAW_CSV_PATH)
    clean_and_save_data(RAW_CSV_PATH, CLEAN_CSV_PATH)


# -------------------------
# APP DATA ACCESS
# -------------------------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the cleaned childcare data from CSV."""
    if not os.path.exists(CLEAN_CSV_PATH) or is_data_stale(CLEAN_CSV_PATH):
        # Try to refresh once; if it fails we rely on existing (possibly older) data.
        try:
            refresh_data()
        except Exception as e:
            # If the clean CSV does not exist at all, we must fail loud.
            if not os.path.exists(CLEAN_CSV_PATH):
                raise RuntimeError(f"Unable to load childcare data: {e}") from e
            # Otherwise we log error via Streamlit and continue with existing file.
            st.warning(
                "Could not refresh childcare data from the City of Toronto API. "
                "Showing previously cached data."
            )

    df = pd.read_csv(CLEAN_CSV_PATH)

    # Ensure expected columns are present; if schema changes we fail clearly.
    required_cols = [
        "location_name",
        "organization_type",
        "address",
        "postal_code",
        "city_ward",
        "phone",
        "building_type",
        "building_name",
        "infant_spaces",
        "toddler_spaces",
        "preschool_spaces",
        "kindergarten_spaces",
        "school_age_spaces",
        "total_spaces",
        "fee_subsidy_available",
        "cwelcc_program",
        "geo_point",
        "region",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in childcare.csv: {missing}")

    return df


def extract_coords(geo: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract latitude and longitude from GeoJSON-like geometry."""
    if pd.isna(geo):
        return None, None

    try:
        g = json.loads(geo)
        return g["coordinates"][1], g["coordinates"][0]
    except Exception:
        return None, None


# -------------------------
# MAP & UI
# -------------------------

def childcare_map(df: pd.DataFrame) -> None:
    st.subheader("Explore Childcare Centres")

    # Filter out rows without coordinates
    df = df.dropna(subset=["lat", "lon"])

    if df.empty:
        st.info("No locations to display on the map with the current filters.")
        return

    # Center map roughly on Toronto
    m = folium.Map(location=[43.7, -79.4], zoom_start=11)

    for _, row in df.iterrows():
        popup_text = f"""
        <b>{row['location_name']}</b><br>
        Address: {row['address']}<br>
        Ward: {row['city_ward']}<br>
        Total Spaces: {row['total_spaces']}
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=popup_text,
        ).add_to(m)

    map_data = st_folium(m, width=700, height=500)

    # If user clicks marker
    last_clicked = map_data.get("last_object_clicked")
    if last_clicked:
        lat = last_clicked.get("lat")
        lon = last_clicked.get("lng")

        if lat is not None and lon is not None:
            selected = df[
                (df["lat"].round(5) == round(lat, 5))
                & (df["lon"].round(5) == round(lon, 5))
            ]

            if not selected.empty:
                row = selected.iloc[0]

                st.markdown("### Selected Location")

                st.write(f"**Name:** {row['location_name']}")
                st.write(f"**Address:** {row['address']}")
                st.write(f"**Phone:** {row['phone']}")
                st.write(f"**Organization Type:** {row['organization_type']}")
                st.write(f"**Total Spaces:** {row['total_spaces']}")
                st.write(f"**Infant Spaces:** {row['infant_spaces']}")
                st.write(f"**Toddler Spaces:** {row['toddler_spaces']}")
                st.write(f"**Preschool Spaces:** {row['preschool_spaces']}")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                180deg,
                #fdfefe 0%,
                #f0f7ff 40%,
                #f7fbff 100%
            );
        }

        .block-container {
            background-color: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            max-width: 1200px;
        }

        section[data-testid="stSidebar"] {
            background-color: #f3f8ff;
            border-right: 1px solid #e6eef8;
        }

        div[data-testid="stMetric"] {
            background-color: #f9fbff;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #e3edf7;
        }

        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        iframe {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# MAIN APP
# -------------------------

def main() -> None:
    st.set_page_config(
        page_title="Toronto Childcare Explorer",
        layout="wide",
        page_icon="🧸",
    )

    inject_css()

    st.title("Toronto Childcare Explorer")

    with st.status("Loading childcare data...", expanded=False) as status:
        try:
            df = load_data()
            status.update(label="Childcare data loaded.", state="complete")
        except Exception as e:
            status.update(label="Failed to load childcare data.", state="error")
            st.error(f"Error loading data: {e}")
            return

    # Extract coordinates (cached within the data cache)
    if "lat" not in df.columns or "lon" not in df.columns:
        df[["lat", "lon"]] = df["geo_point"].apply(
            lambda x: pd.Series(extract_coords(x))
        )

    st.sidebar.header("Filters")

    filtered = df.copy()

    # Region Filter
    region = st.sidebar.multiselect(
        "Region",
        sorted(df["region"].dropna().unique()),
    )
    if region:
        filtered = filtered[filtered["region"].isin(region)]

    # Building Type Filter
    building_type = st.sidebar.multiselect(
        "Building Type",
        sorted(df["building_type"].dropna().unique()),
    )
    if building_type:
        filtered = filtered[filtered["building_type"].isin(building_type)]

    # Organization Type Filter
    organization_type = st.sidebar.multiselect(
        "Organization Type",
        sorted(df["organization_type"].dropna().unique()),
    )
    if organization_type:
        filtered = filtered[filtered["organization_type"].isin(organization_type)]

    # Fee Subsidy Filter
    subsidy_filter = st.sidebar.selectbox(
        "Fee Subsidy Available",
        ["All", "Yes", "No"],
    )
    if subsidy_filter == "Yes":
        filtered = filtered[filtered["fee_subsidy_available"] == "Y"]
    elif subsidy_filter == "No":
        filtered = filtered[filtered["fee_subsidy_available"] == "N"]

    # Map
    st.subheader("Map")
    childcare_map(filtered)



if __name__ == "__main__":
    main()
