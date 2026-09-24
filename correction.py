"""
STAGE 8 — ML ERROR CORRECTION

Input:
    stage6/data/drift_analysis.csv

Stage 7 models:
    models/error_x_model.pkl
    models/error_y_model.pkl
    models/feature_config.json

Output:
    stage8/data/ml_corrected_trajectory.csv

Plots:
    stage8/outputs/stage8_ground_truth_vs_dr.png
    stage8/outputs/stage8_ground_truth_vs_ai.png
    stage8/outputs/stage8_error_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.correction import correct_dead_reckoning


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "stage6"
    / "data"
    / "drift_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "stage8"
    / "data"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ml_corrected_trajectory.csv"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "stage8"
    / "outputs"
)


# ============================================================
# LOAD STAGE 6 DATA
# ============================================================

def load_input():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Stage 6 dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded Stage 6 dataset:")
    print(f"  {INPUT_PATH}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    return df


# ============================================================
# CONVERT STAGE 6 COLUMN NAMES TO ML CONTRACT
# ============================================================

def prepare_stage8_input(df):

    required = [
        "timestamp",
        "ground_truth_x_m",
        "ground_truth_y_m",
        "dr_x_m",
        "dr_y_m",
        "dr_speed_mps",
        "dr_heading_deg",
        "accel_x",
        "accel_y",
        "gyro_z",
        "gnss_available",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Stage 6 dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    result = df.copy()

    # --------------------------------------------------------
    # Stage 6 → Stage 7 naming contract
    # --------------------------------------------------------

    result["x_true"] = result["ground_truth_x_m"]
    result["y_true"] = result["ground_truth_y_m"]

    result["x_dr"] = result["dr_x_m"]
    result["y_dr"] = result["dr_y_m"]

    # Convert DR speed + heading to velocity components.
    heading_rad = np.deg2rad(
        result["dr_heading_deg"]
    )

    result["vx_dr"] = (
        result["dr_speed_mps"]
        * np.cos(heading_rad)
    )

    result["vy_dr"] = (
        result["dr_speed_mps"]
        * np.sin(heading_rad)
    )

    result["heading_dr"] = heading_rad

    result["ax_imu"] = result["accel_x"]
    result["ay_imu"] = result["accel_y"]

    # Stage 7 expects trajectory_id.
    # This dataset contains one trajectory.
    result["trajectory_id"] = "trajectory_001"

    # Stage 7 expects numeric GPS availability.
    result["gps_available"] = (
        result["gnss_available"]
        .astype(int)
    )

    return result


# ============================================================
# CALCULATE ERRORS
# ============================================================

def calculate_errors(df):

    result = df.copy()

    result["original_error_x"] = (
        result["x_true"]
        - result["x_dr"]
    )

    result["original_error_y"] = (
        result["y_true"]
        - result["y_dr"]
    )

    result["original_error_m"] = np.sqrt(
        result["original_error_x"] ** 2
        + result["original_error_y"] ** 2
    )

    result["corrected_error_x"] = (
        result["x_true"]
        - result["x_ai"]
    )

    result["corrected_error_y"] = (
        result["y_true"]
        - result["y_ai"]
    )

    result["corrected_error_m"] = np.sqrt(
        result["corrected_error_x"] ** 2
        + result["corrected_error_y"] ** 2
    )

    return result


# ============================================================
# PLOT 1
# ============================================================

def plot_ground_truth_vs_dr(df):

    path = (
        PLOT_DIR
        / "stage8_ground_truth_vs_dr.png"
    )

    plt.figure(figsize=(9, 7))

    plt.plot(
        df["x_true"],
        df["y_true"],
        linewidth=2,
        label="Ground Truth",
    )

    plt.plot(
        df["x_dr"],
        df["y_dr"],
        linestyle="--",
        linewidth=2,
        label="Dead Reckoning",
    )

    outage = (
        df["gnss_available"] == False
    )

    if outage.any():

        plt.plot(
            df.loc[outage, "x_true"],
            df.loc[outage, "y_true"],
            linewidth=4,
            alpha=0.4,
            label="GNSS Outage",
        )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(
        "Stage 8: Ground Truth vs Dead Reckoning"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================
# PLOT 2
# ============================================================

def plot_ground_truth_vs_ai(df):

    path = (
        PLOT_DIR
        / "stage8_ground_truth_vs_ai.png"
    )

    plt.figure(figsize=(9, 7))

    plt.plot(
        df["x_true"],
        df["y_true"],
        linewidth=2,
        label="Ground Truth",
    )

    plt.plot(
        df["x_ai"],
        df["y_ai"],
        linestyle="--",
        linewidth=2,
        label="AI Corrected",
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(
        "Stage 8: Ground Truth vs AI Corrected Trajectory"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================
# PLOT 3
# ============================================================

def plot_error_comparison(df):

    path = (
        PLOT_DIR
        / "stage8_error_comparison.png"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["timestamp"],
        df["original_error_m"],
        linewidth=1.5,
        label="Original DR Error",
    )

    plt.plot(
        df["timestamp"],
        df["corrected_error_m"],
        linewidth=1.5,
        label="AI Corrected Error",
    )

    outage = (
        df["gnss_available"] == False
    )

    if outage.any():

        outage_time = df.loc[
            outage,
            "timestamp"
        ]

        plt.axvspan(
            outage_time.iloc[0],
            outage_time.iloc[-1],
            alpha=0.2,
            label="GNSS Outage",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title(
        "Stage 8: Original vs AI Corrected Position Error"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================
# VALIDATION
# ============================================================

def validate(df):

    print()
    print("=" * 60)
    print("STAGE 8 VALIDATION")
    print("=" * 60)

    required_outputs = [
        "predicted_error_x",
        "predicted_error_y",
        "x_ai",
        "y_ai",
        "original_error_m",
        "corrected_error_m",
    ]

    missing = [
        column
        for column in required_outputs
        if column not in df.columns
    ]

    if missing:

        print("Output columns: FAIL")
        print("Missing:", missing)
        raise RuntimeError(
            "Stage 8 output validation failed."
        )

    print("Output columns: PASS")

    if df[
        ["predicted_error_x", "predicted_error_y"]
    ].isna().any().any():

        print("Prediction values: FAIL")

        raise RuntimeError(
            "NaN values found in ML predictions."
        )

    print("Prediction values: PASS")

    if df[
        ["x_ai", "y_ai"]
    ].isna().any().any():

        print("AI corrected coordinates: FAIL")

        raise RuntimeError(
            "NaN values found in corrected coordinates."
        )

    print("AI corrected coordinates: PASS")

    original_mean = df["original_error_m"].mean()
    corrected_mean = df["corrected_error_m"].mean()

    original_rmse = np.sqrt(
        np.mean(
            df["original_error_m"] ** 2
        )
    )

    corrected_rmse = np.sqrt(
        np.mean(
            df["corrected_error_m"] ** 2
        )
    )

    print()
    print("ERROR METRICS")
    print("-" * 60)

    print(
        f"Original Mean Error:  "
        f"{original_mean:.4f} m"
    )

    print(
        f"Corrected Mean Error: "
        f"{corrected_mean:.4f} m"
    )

    print(
        f"Original RMSE:         "
        f"{original_rmse:.4f} m"
    )

    print(
        f"Corrected RMSE:        "
        f"{corrected_rmse:.4f} m"
    )

    if original_rmse > 0:

        improvement = (
            (original_rmse - corrected_rmse)
            / original_rmse
        ) * 100

        print(
            f"RMSE Improvement:      "
            f"{improvement:.2f}%"
        )

    print()
    print("Stage 8 validation: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("STAGE 8 — ML ERROR CORRECTION")
    print("=" * 60)
    print()

    df = load_input()

    print()
    print("Converting Stage 6 columns to Stage 7 ML format...")

    df = prepare_stage8_input(df)

    print("Column conversion: PASS")

    print()
    print("Running Stage 7 ML error prediction...")

    corrected = correct_dead_reckoning(df)

    print("ML prediction: PASS")

    corrected = calculate_errors(
        corrected
    )

    corrected.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_ground_truth_vs_dr(
        corrected
    )

    plot_ground_truth_vs_ai(
        corrected
    )

    plot_error_comparison(
        corrected
    )

    validate(corrected)

    print()
    print("=" * 60)
    print("STAGE 8 OUTPUT")
    print("=" * 60)

    print()
    print(
        f"CSV:\n  {OUTPUT_PATH}"
    )

    print()
    print("Plots:")

    print(
        f"  {PLOT_DIR / 'stage8_ground_truth_vs_dr.png'}"
    )

    print(
        f"  {PLOT_DIR / 'stage8_ground_truth_vs_ai.png'}"
    )

    print(
        f"  {PLOT_DIR / 'stage8_error_comparison.png'}"
    )

    print()
    print("STAGE 8 RESULT: PASS")


if __name__ == "__main__":
    main()