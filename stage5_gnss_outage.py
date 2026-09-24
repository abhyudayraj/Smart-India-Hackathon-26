"""
STAGE 5 — Simulated GNSS Outage

Input:
    stage4/data/dead_reckoning.csv

Output:
    stage5/data/gnss_outage.csv
    stage5/outputs/stage5_gnss_outage.png
    stage5/outputs/stage5_gnss_status.png

Purpose:
    Simulate a GNSS-denied section of the route based on cumulative
    ground-truth distance travelled.

Important:
    Dead Reckoning coordinates (dr_x_m, dr_y_m) are NEVER modified.
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "stage4/data/dead_reckoning.csv"

OUTPUT_CSV_PATH = "stage5/data/gnss_outage.csv"
OUTPUT_TRAJECTORY_PLOT = "stage5/outputs/stage5_gnss_outage.png"
OUTPUT_STATUS_PLOT = "stage5/outputs/stage5_gnss_status.png"

# GNSS outage interval in cumulative ground-truth distance
OUTAGE_START_M = 300.0
OUTAGE_END_M = 700.0


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",

    # Ground truth
    "ground_truth_x_m",
    "ground_truth_y_m",
    "ground_speed",
    "ground_heading",
    "ground_acceleration",
    "ground_angular_velocity",
    "ground_latitude",
    "ground_longitude",

    # GPS information
    "gps_x",
    "gps_y",
    "gps_latitude",
    "gps_longitude",

    # Sensor information
    "accel_x",
    "accel_y",
    "gyro_z",

    # Dead reckoning
    "dr_x_m",
    "dr_y_m",
]


# ============================================================
# LOAD STAGE 4 DATA
# ============================================================

def load_input(path):
    if not os.path.exists(path):
        print(f"ERROR: Input file not found: {path}")
        print()
        print("Make sure Stage 4 has been run and the file exists at:")
        print("stage4/data/dead_reckoning.csv")
        sys.exit(1)

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        print("ERROR: Input file is missing required column(s):")

        for column in missing:
            print(f"  - {column}")

        sys.exit(1)

    return df


# ============================================================
# COMPUTE CUMULATIVE GROUND-TRUTH DISTANCE
# ============================================================

def compute_cumulative_distance(df):
    x = df["ground_truth_x_m"].to_numpy(dtype=float)
    y = df["ground_truth_y_m"].to_numpy(dtype=float)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    # No movement before the first sample
    dx[0] = 0.0
    dy[0] = 0.0

    distance_increment = np.sqrt(dx**2 + dy**2)

    cumulative_distance_m = np.cumsum(distance_increment)

    return cumulative_distance_m


# ============================================================
# SIMULATE GNSS OUTAGE
# ============================================================

def simulate_gnss_outage(df):
    df = df.copy()

    # Calculate distance travelled along the ground-truth trajectory
    df["cumulative_distance_m"] = compute_cumulative_distance(df)

    # Identify the simulated GNSS-denied section
    outage_mask = (
        (df["cumulative_distance_m"] >= OUTAGE_START_M)
        & (df["cumulative_distance_m"] <= OUTAGE_END_M)
    )

    # GNSS is available everywhere except the outage section
    df["gnss_available"] = ~outage_mask

    # GNSS position is available only when GNSS is available
    df["gnss_x_m"] = np.where(
        df["gnss_available"],
        df["ground_truth_x_m"],
        np.nan,
    )

    df["gnss_y_m"] = np.where(
        df["gnss_available"],
        df["ground_truth_y_m"],
        np.nan,
    )

    # Navigation mode
    df["navigation_mode"] = np.where(
        df["gnss_available"],
        "GNSS + DR",
        "Dead Reckoning",
    )

    return df


# ============================================================
# TRAJECTORY PLOT
# ============================================================

def make_trajectory_plot(df, path):
    fig, ax = plt.subplots(figsize=(9, 7))

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ax.plot(
        df["ground_truth_x_m"],
        df["ground_truth_y_m"],
        color="tab:green",
        linewidth=2,
        label="Ground Truth",
    )

    # --------------------------------------------------------
    # Dead reckoning
    # --------------------------------------------------------

    ax.plot(
        df["dr_x_m"],
        df["dr_y_m"],
        color="tab:red",
        linewidth=2,
        linestyle="--",
        label="Dead Reckoning",
    )

    # --------------------------------------------------------
    # GNSS available positions
    # --------------------------------------------------------

    ax.scatter(
        df["gnss_x_m"],
        df["gnss_y_m"],
        color="tab:blue",
        s=12,
        label="GNSS Available",
        zorder=3,
    )

    # --------------------------------------------------------
    # Highlight outage section
    # --------------------------------------------------------

    outage_mask = ~df["gnss_available"]

    if outage_mask.any():

        outage_x = df.loc[
            outage_mask,
            "ground_truth_x_m"
        ]

        outage_y = df.loc[
            outage_mask,
            "ground_truth_y_m"
        ]

        ax.plot(
            outage_x,
            outage_y,
            color="orange",
            linewidth=5,
            alpha=0.4,
            label="GNSS Outage Section",
            zorder=2,
        )

        # First outage sample
        first_idx = df.loc[outage_mask].index[0]

        # Last outage sample
        last_idx = df.loc[outage_mask].index[-1]

        # Outage start marker
        ax.scatter(
            df.loc[first_idx, "ground_truth_x_m"],
            df.loc[first_idx, "ground_truth_y_m"],
            color="black",
            marker="x",
            s=100,
            zorder=4,
            label="Outage Start",
        )

        # Outage end marker
        ax.scatter(
            df.loc[last_idx, "ground_truth_x_m"],
            df.loc[last_idx, "ground_truth_y_m"],
            color="purple",
            marker="x",
            s=100,
            zorder=4,
            label="Outage End",
        )

    # --------------------------------------------------------
    # Plot formatting
    # --------------------------------------------------------

    ax.set_title(
        "Stage 5: Ground Truth vs Dead Reckoning with GNSS Outage"
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    ax.legend(loc="best")

    ax.set_aspect("equal", adjustable="datalim")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# GNSS STATUS PLOT
# ============================================================

def make_status_plot(df, path):
    fig, ax = plt.subplots(figsize=(9, 3.5))

    status = df["gnss_available"].astype(int)

    # GNSS availability
    ax.plot(
        df["cumulative_distance_m"],
        status,
        color="tab:blue",
        linewidth=2,
        drawstyle="steps-post",
    )

    ax.fill_between(
        df["cumulative_distance_m"],
        status,
        step="post",
        alpha=0.15,
        color="tab:blue",
    )

    # Outage zone
    ax.axvspan(
        OUTAGE_START_M,
        OUTAGE_END_M,
        color="orange",
        alpha=0.2,
        label="Outage Zone",
    )

    ax.set_title(
        "Stage 5: GNSS Availability vs Distance Travelled"
    )

    ax.set_xlabel("Cumulative Distance (m)")
    ax.set_ylabel("GNSS Status")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        "Unavailable",
        "Available",
    ])

    ax.set_ylim(-0.2, 1.2)

    ax.legend(loc="upper right")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VALIDATION
# ============================================================

def validate_output(df):
    print()
    print("==============================")
    print("STAGE 5 VALIDATION")
    print("==============================")

    # --------------------------------------------------------
    # Check DR coordinates were not modified
    # --------------------------------------------------------

    if "dr_x_m" not in df.columns or "dr_y_m" not in df.columns:
        print("FAIL: DR coordinate columns missing.")
        return False

    if df["dr_x_m"].isna().any() or df["dr_y_m"].isna().any():
        print("WARNING: DR coordinates contain NaN values.")

    # --------------------------------------------------------
    # Check outage
    # --------------------------------------------------------

    outage_mask = ~df["gnss_available"]

    if not outage_mask.any():
        print("FAIL: No GNSS outage samples found.")
        return False

    outage_distances = df.loc[
        outage_mask,
        "cumulative_distance_m"
    ]

    actual_start = outage_distances.iloc[0]
    actual_end = outage_distances.iloc[-1]

    print(
        f"Requested outage: "
        f"{OUTAGE_START_M:.2f} m to {OUTAGE_END_M:.2f} m"
    )

    print(
        f"Actual outage: "
        f"{actual_start:.2f} m to {actual_end:.2f} m"
    )

    # --------------------------------------------------------
    # Check GNSS coordinates are missing during outage
    # --------------------------------------------------------

    outage_gnss_x_missing = df.loc[
        outage_mask,
        "gnss_x_m"
    ].isna().all()

    outage_gnss_y_missing = df.loc[
        outage_mask,
        "gnss_y_m"
    ].isna().all()

    if not outage_gnss_x_missing:
        print("FAIL: gnss_x_m is not fully missing during outage.")
        return False

    if not outage_gnss_y_missing:
        print("FAIL: gnss_y_m is not fully missing during outage.")
        return False

    print("GNSS coordinates during outage: PASS")

    # --------------------------------------------------------
    # Check GNSS coordinates outside outage
    # --------------------------------------------------------

    available_mask = df["gnss_available"]

    if available_mask.any():

        x_match = np.allclose(
            df.loc[available_mask, "gnss_x_m"].to_numpy(),
            df.loc[available_mask, "ground_truth_x_m"].to_numpy(),
            equal_nan=False,
        )

        y_match = np.allclose(
            df.loc[available_mask, "gnss_y_m"].to_numpy(),
            df.loc[available_mask, "ground_truth_y_m"].to_numpy(),
            equal_nan=False,
        )

        if not x_match or not y_match:
            print(
                "FAIL: GNSS coordinates do not match "
                "ground truth outside outage."
            )
            return False

    print("GNSS coordinates outside outage: PASS")

    # --------------------------------------------------------
    # Check navigation modes
    # --------------------------------------------------------

    correct_outage_mode = (
        df.loc[outage_mask, "navigation_mode"]
        == "Dead Reckoning"
    ).all()

    correct_available_mode = (
        df.loc[available_mask, "navigation_mode"]
        == "GNSS + DR"
    ).all()

    if not correct_outage_mode:
        print("FAIL: Incorrect navigation mode during outage.")
        return False

    if not correct_available_mode:
        print("FAIL: Incorrect navigation mode outside outage.")
        return False

    print("Navigation modes: PASS")

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    print()
    print("Stage 5 validation: PASS")

    return True


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    total_distance = df[
        "cumulative_distance_m"
    ].iloc[-1]

    outage_mask = ~df["gnss_available"]

    if outage_mask.any():

        outage_start_actual = df.loc[
            outage_mask,
            "cumulative_distance_m"
        ].iloc[0]

        outage_end_actual = df.loc[
            outage_mask,
            "cumulative_distance_m"
        ].iloc[-1]

    else:

        outage_start_actual = float("nan")
        outage_end_actual = float("nan")

    available_count = int(
        df["gnss_available"].sum()
    )

    unavailable_count = int(
        (~df["gnss_available"]).sum()
    )

    mode_counts = df[
        "navigation_mode"
    ].value_counts()

    gnss_dr_count = int(
        mode_counts.get("GNSS + DR", 0)
    )

    dr_only_count = int(
        mode_counts.get("Dead Reckoning", 0)
    )

    print()
    print("==============================")
    print("STAGE 5: GNSS OUTAGE SIMULATION")
    print("==============================")

    print(
        f"Total route distance: "
        f"{total_distance:.2f} m"
    )

    print()

    print(
        f"Requested outage: "
        f"{OUTAGE_START_M:.2f} m to "
        f"{OUTAGE_END_M:.2f} m"
    )

    print(
        f"Actual outage: "
        f"{outage_start_actual:.2f} m to "
        f"{outage_end_actual:.2f} m"
    )

    print()

    print(
        f"GNSS available samples: "
        f"{available_count}"
    )

    print(
        f"GNSS unavailable samples: "
        f"{unavailable_count}"
    )

    print()

    print("Navigation modes:")

    print(
        f"GNSS + DR: "
        f"{gnss_dr_count} samples"
    )

    print(
        f"Dead Reckoning: "
        f"{dr_only_count} samples"
    )

    print()

    print("Output saved to:")
    print(f"  {OUTPUT_CSV_PATH}")

    print()

    print("Plots saved to:")
    print(f"  {OUTPUT_TRAJECTORY_PLOT}")
    print(f"  {OUTPUT_STATUS_PLOT}")


# ============================================================
# MAIN
# ============================================================

def main():

    # Create Stage 5 directories
    os.makedirs(
        "stage5/data",
        exist_ok=True
    )

    os.makedirs(
        "stage5/outputs",
        exist_ok=True
    )

    # Load Stage 4 output
    df = load_input(INPUT_PATH)

    print("Loaded Stage 4 dataset:")
    print(f"  {INPUT_PATH}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Simulate GNSS outage
    df = simulate_gnss_outage(df)

    # Save output
    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False
    )

    # Generate plots
    make_trajectory_plot(
        df,
        OUTPUT_TRAJECTORY_PLOT
    )

    make_status_plot(
        df,
        OUTPUT_STATUS_PLOT
    )

    # Validate
    validation_passed = validate_output(df)

    # Print summary
    print_summary(df)

    if not validation_passed:
        print()
        print("STAGE 5 RESULT: FAIL")
        sys.exit(1)

    print()
    print("STAGE 5 RESULT: PASS")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()