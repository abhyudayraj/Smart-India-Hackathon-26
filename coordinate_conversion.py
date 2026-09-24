"""
Stage 3 — GPS (lat/lon) -> local metric X/Y coordinates.

Input:
    stage2/data/raw/trajectory.csv

Stage 2 GPS columns:
    timestamp       -> seconds
    gps_latitude    -> WGS84 latitude, degrees
    gps_longitude   -> WGS84 longitude, degrees

Output:
    stage3/data/processed/trajectory_xy.csv
    stage3/outputs/plots/gps_xy.png

Scope:
    - Read GPS latitude, longitude and timestamp.
    - Convert GPS coordinates to local metric X/Y coordinates.
    - Use the first valid GPS fix as the local origin (0, 0).
    - Preserve timestamps in seconds.
    - Preserve rows where GPS is unavailable.
    - Plot the resulting GPS trajectory.

Not in scope:
    - Dead reckoning / IMU integration
    - Machine learning
    - Map matching
    - GNSS/IMU fusion
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Stage 2 input column names
# ---------------------------------------------------------------------------

TIME_COL = "timestamp"
LAT_COL = "gps_latitude"
LON_COL = "gps_longitude"


def load_gps_data(csv_path: str) -> pd.DataFrame:
    """
    Load timestamp, latitude and longitude from the Stage 2 trajectory.

    Timestamp is already in seconds, so no milliseconds-to-seconds
    conversion is performed.
    """

    df = pd.read_csv(
        csv_path,
        usecols=[TIME_COL, LAT_COL, LON_COL]
    )

    df = df.rename(
        columns={
            TIME_COL: "timestamp",
            LAT_COL: "lat_deg",
            LON_COL: "lon_deg",
        }
    )

    return df


def find_origin(df: pd.DataFrame) -> tuple[float, float]:
    """
    Find the first valid GPS fix and use it as the local coordinate origin.
    """

    valid = df.dropna(subset=["lat_deg", "lon_deg"])

    if valid.empty:
        raise ValueError("No valid GPS fixes found in input data.")

    first = valid.iloc[0]

    return (
        float(first["lat_deg"]),
        float(first["lon_deg"])
    )


def build_transformer(
    origin_lat: float,
    origin_lon: float
) -> Transformer:
    """
    Build an Azimuthal Equidistant projection centred on the first
    valid GPS fix.

    Input:
        WGS84 latitude/longitude

    Output:
        Local metric X/Y coordinates in metres
    """

    aeqd_proj4 = (
        f"+proj=aeqd "
        f"+lat_0={origin_lat} "
        f"+lon_0={origin_lon} "
        f"+datum=WGS84 "
        f"+units=m "
        f"+no_defs"
    )

    return Transformer.from_crs(
        "EPSG:4326",
        aeqd_proj4,
        always_xy=True
    )


def convert_to_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert GPS latitude/longitude into local metric X/Y coordinates.

    GPS outage rows are preserved. Their x_m and y_m values become NaN.
    """

    origin_lat, origin_lon = find_origin(df)

    transformer = build_transformer(
        origin_lat,
        origin_lon
    )

    out = df.copy()

    valid_mask = (
        out["lat_deg"].notna()
        & out["lon_deg"].notna()
    )

    x = np.full(len(out), np.nan)
    y = np.full(len(out), np.nan)

    lon_values = out.loc[
        valid_mask,
        "lon_deg"
    ].to_numpy()

    lat_values = out.loc[
        valid_mask,
        "lat_deg"
    ].to_numpy()

    x_values, y_values = transformer.transform(
        lon_values,
        lat_values
    )

    x[valid_mask.to_numpy()] = x_values
    y[valid_mask.to_numpy()] = y_values

    out["x_m"] = x
    out["y_m"] = y

    print(
        f"Origin (first valid GPS fix): "
        f"lat={origin_lat:.7f}, "
        f"lon={origin_lon:.7f}"
    )

    print(
        "Origin converted to local coordinates: "
        "(x=0.000 m, y=0.000 m)"
    )

    return out


def plot_trajectory(
    df: pd.DataFrame,
    out_path: str
) -> None:
    """
    Plot local X/Y GPS trajectory.
    """

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    valid = df.dropna(
        subset=["x_m", "y_m"]
    )

    ax.plot(
        valid["x_m"],
        valid["y_m"],
        linewidth=1,
        color="tab:blue"
    )

    if not valid.empty:

        ax.scatter(
            valid["x_m"].iloc[0],
            valid["y_m"].iloc[0],
            color="green",
            zorder=5,
            label="Start (0, 0)"
        )

        ax.scatter(
            valid["x_m"].iloc[-1],
            valid["y_m"].iloc[-1],
            color="red",
            zorder=5,
            label="End"
        )

    ax.set_xlabel(
        "X (metres, East-ish)"
    )

    ax.set_ylabel(
        "Y (metres, North-ish)"
    )

    ax.set_title(
        "GPS Trajectory in Local Metric Coordinates (AEQD)"
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend()

    fig.tight_layout()

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        out_path,
        dpi=150
    )

    plt.close(fig)

    print(
        f"Plot written to {out_path}"
    )


def validate_output(df: pd.DataFrame) -> None:
    """
    Perform basic Stage 3 validation.
    """

    print()
    print("=" * 70)
    print("STAGE 3 VALIDATION")
    print("=" * 70)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # Check timestamp
    timestamps = df["timestamp"]

    if timestamps.is_monotonic_increasing:
        print(
            "[PASS] Timestamps are increasing"
        )
    else:
        print(
            "[FAIL] Timestamps are not increasing"
        )

    # Check timestamp range
    print(
        f"Timestamp range: "
        f"{timestamps.iloc[0]:.2f}s -> "
        f"{timestamps.iloc[-1]:.2f}s"
    )

    # Check coordinate columns
    if "x_m" in df.columns and "y_m" in df.columns:
        print(
            "[PASS] x_m and y_m columns created"
        )
    else:
        print(
            "[FAIL] x_m/y_m columns missing"
        )
        return

    # Count valid GPS coordinates
    valid_xy = df.dropna(
        subset=["x_m", "y_m"]
    )

    missing_xy = len(df) - len(valid_xy)

    print(
        f"Valid GPS coordinate rows: {len(valid_xy)}"
    )

    print(
        f"Missing GPS coordinate rows: {missing_xy}"
    )

    # Check first valid point is approximately origin
    if not valid_xy.empty:

        first_x = valid_xy["x_m"].iloc[0]
        first_y = valid_xy["y_m"].iloc[0]

        if (
            abs(first_x) < 1e-6
            and abs(first_y) < 1e-6
        ):
            print(
                "[PASS] First valid GPS fix is at (0, 0)"
            )
        else:
            print(
                "[FAIL] First valid GPS fix is not at (0, 0)"
            )

    # Check expected Stage 2 row count
    if len(df) == 1200:
        print(
            "[PASS] Row count = 1200"
        )
    else:
        print(
            f"[WARNING] Expected 1200 rows, found {len(df)}"
        )

    print()
    print("Stage 3 coordinate conversion validation complete.")


def main():

    # -----------------------------------------------------------------------
    # Paths are relative to the SIH project root:
    #
    # C:\Users\KIIT\Desktop\SIH
    # -----------------------------------------------------------------------

    input_csv = (
        "stage2/data/raw/trajectory.csv"
    )

    output_csv = (
        "stage3/data/processed/trajectory_xy.csv"
    )

    output_plot = (
        "stage3/outputs/plots/gps_xy.png"
    )

    # Check input exists
    if not Path(input_csv).exists():

        raise FileNotFoundError(
            f"Input file not found: {input_csv}"
        )

    print(
        f"Reading Stage 2 dataset:\n"
        f"  {input_csv}"
    )

    # Load
    df = load_gps_data(
        input_csv
    )

    print(
        f"Loaded {len(df)} rows."
    )

    # Convert
    df_xy = convert_to_xy(
        df
    )

    # Create output directory
    Path(output_csv).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save converted data
    df_xy.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Converted trajectory written to:\n"
        f"  {output_csv}"
    )

    # Plot
    plot_trajectory(
        df_xy,
        output_plot
    )

    # Validate
    validate_output(
        df_xy
    )


if __name__ == "__main__":
    main()