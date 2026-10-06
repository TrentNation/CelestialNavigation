#!/usr/bin/env python3
"""
Bulk night-sky star chart generator.

Reads a CSV of (id, datetime_utc, latitude, longitude) rows and generates one
unlabeled star chart PNG per row, showing the real naked-eye sky (~8,900 stars,
mag <= 6.5) visible at that exact time and place. A companion CSV is written
for each chart with the plotted position and source catalog coordinates of
every visible star.

Usage:
    python3 generate_charts.py input.csv output_folder/

Input CSV columns (header required):
    id            - used as the output filename (no extension)
    datetime_utc  - ISO 8601, e.g. 2026-07-23T22:00:00  (assumed UTC)
    latitude      - decimal degrees, -90 to 90
    longitude     - decimal degrees, -180 to 180 (east positive)

Everything needed (ephemeris + star catalog) is bundled locally in data/ -
no network access is required at run time.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import csv
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos  # noqa: F401 (not used, kept for reference)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BSP_PATH = os.path.join(DATA_DIR, "de421.bsp")
STARS_PATH = os.path.join(DATA_DIR, "stars_slim.csv")

MAG_LIMIT = 4  # naked-eye limit already baked into stars_slim.csv


def load_catalog():
    stars_df = pd.read_csv(STARS_PATH)
    eph = load(BSP_PATH)
    ts = load.timescale()
    earth = eph["earth"]
    # Build one big Star object for vectorized alt/az computation
    star_obj = Star(ra_hours=stars_df["ra"].values, dec_degrees=stars_df["dec"].values)
    right_ascension = stars_df["ra"].values
    declination = stars_df["dec"].values
    mags = stars_df["mag"].values
    return eph, ts, earth, star_obj, mags, right_ascension, declination


def render_chart(
    earth, ts, star_obj, mags, right_ascension, declination,
    dt_utc, lat, lon, out_path, size_px=1200,
):
    t = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day,
               dt_utc.hour, dt_utc.minute, dt_utc.second)
    observer = earth + wgs84.latlon(lat, lon)
    astrometric = observer.at(t).observe(star_obj)
    alt, az, _ = astrometric.apparent().altaz()

    alt_deg = alt.degrees
    az_deg = az.degrees
    visible = alt_deg > 0

    alt_v = alt_deg[visible]
    az_v = az_deg[visible]
    mag_v = mags[visible]
    right_ascension_v = right_ascension[visible]
    declination_v = declination[visible]

    # Marker size: brighter (lower mag) = bigger dot
    sizes = np.clip((MAG_LIMIT - mag_v) ** 2.2 * 1.1 + 0.3, 0.3, None)

    theta = np.radians(az_v)
    r = 90 - alt_v  # zenith at center (r=0), horizon at edge (r=90)

    fig = plt.figure(figsize=(8, 8), dpi=size_px / 8)
    ax = fig.add_subplot(111, projection="polar")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.scatter(theta, r, s=sizes, c="white", linewidths=0)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # azimuth increases clockwise (N->E->S->W)
    ax.set_rlim(0, 90)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    plt.tight_layout(pad=0)

    # Embed the datetime/lat/lon used to generate this chart as PNG metadata
    # (readable later with e.g. `exiftool file.png` or PIL's Image.text / .info)
    png_metadata = {
        "Datetime_UTC": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Latitude": f"{lat:.6f}",
        "Longitude": f"{lon:.6f}",
        "Description": (
            f"Star chart for lat={lat:.6f}, lon={lon:.6f}, "
            f"time={dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        ),
        "Software": "generate_charts.py (skyfield + HYG catalog)",
    }
    fig.savefig(
        out_path,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0,
        metadata=png_metadata,
    )
    plt.close(fig)

    star_csv_path = os.path.splitext(out_path)[0] + ".csv"
    star_data = pd.DataFrame({
        # pos is the same polar position used by matplotlib: azimuth in
        # degrees clockwise from north and radial distance from zenith.
        "pos": [f"azimuth={azimuth:.6f}, radial={90 - altitude:.6f}"
                for azimuth, altitude in zip(az_v, alt_v)],
        "brightness": mag_v,
        "declination": declination_v,
        "right_ascension": right_ascension_v,
        "altitude_degrees": alt_v,
        "azimuth_degrees": az_v,
    })
    star_data.to_csv(star_csv_path, index=False)
    return star_csv_path



def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_charts.py input.csv output_folder/")
        sys.exit(1)

    input_csv = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    rows = pd.read_csv(input_csv)
    rows["datetime_utc"] = pd.to_datetime(rows["datetime_utc"])

    print(f"Loading star catalog and ephemeris...")
    eph, ts, earth, star_obj, mags, right_ascension, declination = load_catalog()

    print(f"Generating {len(rows)} charts...")
    for i, row in rows.iterrows():
        out_path = os.path.join(out_dir, f"{row['id']}.png")
        star_csv_path = render_chart(
            earth, ts, star_obj, mags, right_ascension, declination,
            row["datetime_utc"].to_pydatetime(),
            float(row["latitude"]),
            float(row["longitude"]),
            out_path,
        )
        print(f"  [{i+1}/{len(rows)}] {row['id']} -> {out_path}, {star_csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
