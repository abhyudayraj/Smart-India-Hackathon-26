"""
STAGE 7 — Feature Engineering

Converts the actual Stage 6 dataset into causal ML features.

Input:
    stage6/data/drift_analysis.csv

Features use only information available during live navigation.

Ground truth and actual DR position are NEVER used as features.

Targets:
    error_x = ground_truth_x_m - dr_x_m
    error_y = ground_truth_y_m - dr_y_m
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# FORBIDDEN COLUMNS
# ============================================================

FORBIDDEN_COLUMNS = [
    "ground_truth_x_m",
    "ground_truth_y_m",
    "dr_x_m",
    "dr_y_m",
    "position_error_m",
    "error_x_m",
    "error_y_m",
    "dr_error_m",
]


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "ax_imu",
    "ay_imu",
    "gyro_z",
    "vx_dr",
    "vy_dr",
    "heading_dr",
    "speed_dr",
    "acceleration_magnitude",
    "gyro_magnitude",
    "time_since_last_gps",
]


# ============================================================
# INPUT COLUMNS FROM STAGE 6
# ============================================================

REQUIRED_INPUT_COLUMNS = [
    "timestamp",
    "accel_x",
    "accel_y",
    "gyro_z",
    "dr_speed_mps",
    "dr_heading_deg",
    "gnss_available",
]


# ============================================================
# CREATE TIME SINCE LAST GNSS FIX
# ============================================================

def _time_since_last_gnss(group: pd.DataFrame) -> pd.Series:

    t = group["timestamp"].to_numpy(dtype=float)

    gnss = group["gnss_available"].to_numpy()

    last_gnss_time = np.full(
        len(group),
        np.nan,
        dtype=float,
    )

    running_last = np.nan

    for i in range(len(group)):

        if bool(gnss[i]):

            running_last = t[i]

        last_gnss_time[i] = running_last

    since = t - last_gnss_time

    # If no GNSS fix has appeared yet, use time from
    # beginning of the trajectory.
    since = np.where(
        np.isnan(since),
        t - t[0],
        since,
    )

    return pd.Series(
        since,
        index=group.index,
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    missing = [
        c
        for c in REQUIRED_INPUT_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "prepare_features: missing required columns: "
            f"{missing}"
        )

    out = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Raw IMU features
    # --------------------------------------------------------

    out["ax_imu"] = (
        df["accel_x"].astype(float)
    )

    out["ay_imu"] = (
        df["accel_y"].astype(float)
    )

    out["gyro_z"] = (
        df["gyro_z"].astype(float)
    )

    # --------------------------------------------------------
    # DR speed and heading
    # --------------------------------------------------------

    out["speed_dr"] = (
        df["dr_speed_mps"].astype(float)
    )

    # Convert degrees → radians.
    # This keeps the feature convention consistent with
    # the original Stage 7 synthetic implementation.
    out["heading_dr"] = np.deg2rad(
        df["dr_heading_deg"].astype(float)
    )

    # --------------------------------------------------------
    # DR velocity components
    # --------------------------------------------------------

    out["vx_dr"] = (
        out["speed_dr"]
        * np.cos(out["heading_dr"])
    )

    out["vy_dr"] = (
        out["speed_dr"]
        * np.sin(out["heading_dr"])
    )

    # --------------------------------------------------------
    # Derived sensor features
    # --------------------------------------------------------

    out["acceleration_magnitude"] = np.sqrt(
        out["ax_imu"] ** 2
        + out["ay_imu"] ** 2
    )

    out["gyro_magnitude"] = (
        out["gyro_z"].abs()
    )

    # --------------------------------------------------------
    # Time since last GNSS
    # --------------------------------------------------------

    # Your current project has one trajectory.
    # Therefore calculate directly across the dataframe.
    out["time_since_last_gps"] = (
        _time_since_last_gnss(df.reset_index(drop=True))
        .to_numpy()
    )

    # Restore original index.
    out["time_since_last_gps"] = pd.Series(
        out["time_since_last_gps"].to_numpy(),
        index=df.index,
    )

    # --------------------------------------------------------
    # Exact feature ordering
    # --------------------------------------------------------

    out = out[FEATURE_COLUMNS]

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    forbidden_present = [
        c
        for c in FORBIDDEN_COLUMNS
        if c in out.columns
    ]

    if forbidden_present:

        raise AssertionError(
            "Forbidden columns leaked into features: "
            f"{forbidden_present}"
        )

    if out.isna().any().any():

        missing_features = [
            c
            for c in out.columns
            if out[c].isna().any()
        ]

        raise ValueError(
            "NaN values found in features: "
            f"{missing_features}"
        )

    return out


# ============================================================
# CREATE SUPERVISED TARGETS
# ============================================================

def make_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "ground_truth_x_m",
        "ground_truth_y_m",
        "dr_x_m",
        "dr_y_m",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "make_targets: missing required columns: "
            f"{missing}"
        )

    targets = pd.DataFrame(
        index=df.index
    )

    targets["error_x"] = (
        df["ground_truth_x_m"]
        - df["dr_x_m"]
    )

    targets["error_y"] = (
        df["ground_truth_y_m"]
        - df["dr_y_m"]
    )

    return targets