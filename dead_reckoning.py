"""
STAGE 4 — Dead Reckoning

Input:
    stage2/data/raw/trajectory.csv
    stage3/data/processed/trajectory_xy.csv

Output:
    stage4/data/dead_reckoning.csv
    stage4/outputs/stage4_ground_truth_vs_dr.png

Purpose:
    Build a baseline Dead Reckoning trajectory using simulated vehicle
    speed and heading, with controlled sensor errors.

Important behavior:
    1. DR starts at (0, 0).
    2. Ground truth exists for the complete trajectory.
    3. GNSS is unavailable from 60–90 seconds.
    4. DR continues estimating position during the outage.
    5. DR error increases during the outage.
    6. GNSS becomes available again after 90 seconds.
    7. GNSS availability does NOT affect DR integration.
    8. Ground truth is used only for evaluation.

Machine learning is NOT used in Stage 4.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

STAGE2_CSV = "stage2/data/raw/trajectory.csv"
STAGE3_CSV = "stage3/data/processed/trajectory_xy.csv"

OUTPUT_CSV = "stage4/data/dead_reckoning.csv"
OUTPUT_PLOT = "stage4/outputs/stage4_ground_truth_vs_dr.png"

# GNSS outage
OUTAGE_START = 60.0
OUTAGE_END = 90.0

# Controlled DR sensor errors
SPEED_BIAS_MPS = 0.05
SPEED_NOISE_STD_MPS = 0.02

HEADING_BIAS_DEG = 0.5
HEADING_NOISE_STD_DEG = 0.15

RANDOM_SEED = 42


# ============================================================
# STEP 1: LOAD STAGE 2
# ============================================================

def load_stage2():

    if not os.path.exists(STAGE2_CSV):
        raise FileNotFoundError(
            f"Stage 2 file not found:\n{STAGE2_CSV}"
        )

    df = pd.read_csv(STAGE2_CSV)

    print("Loaded Stage 2 dataset:")
    print(f"  {STAGE2_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    required = [
        "timestamp",
        "ground_speed",
        "ground_heading",
        "gps_available",
        "accel_x",
        "accel_y",
        "gyro_z"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Stage 2 columns: {missing}"
        )

    return df


# ============================================================
# STEP 2: LOAD STAGE 3
# ============================================================

def load_stage3():

    if not os.path.exists(STAGE3_CSV):
        raise FileNotFoundError(
            f"Stage 3 file not found:\n{STAGE3_CSV}"
        )

    xy = pd.read_csv(STAGE3_CSV)

    print("\nLoaded Stage 3 coordinate dataset:")
    print(f"  {STAGE3_CSV}")
    print(f"  Rows: {len(xy)}")

    required = [
        "timestamp",
        "x_m",
        "y_m"
    ]

    missing = [
        col
        for col in required
        if col not in xy.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Stage 3 columns: {missing}"
        )

    return xy


# ============================================================
# STEP 3: VERIFY STAGE 2 + STAGE 3
# ============================================================

def verify_stage_data(stage2, stage3):

    if len(stage2) != len(stage3):
        raise ValueError(
            "Stage 2 and Stage 3 row counts do not match."
        )

    timestamp_difference = np.abs(
        stage2["timestamp"].to_numpy(dtype=float)
        -
        stage3["timestamp"].to_numpy(dtype=float)
    )

    if not np.isfinite(timestamp_difference).all():
        raise ValueError(
            "Invalid timestamps detected."
        )

    if np.max(timestamp_difference) > 1e-6:
        raise ValueError(
            "Stage 2 and Stage 3 timestamps do not match."
        )

    print("\nStage 2 + Stage 3 combination: PASS")
    print(f"  Stage 2 rows: {len(stage2)}")
    print(f"  Stage 3 rows: {len(stage3)}")
    print(f"  Combined rows: {len(stage2)}")

    # Stage 3 is expected to contain NaN coordinates during
    # the simulated GNSS outage.
    stage3_nan_x = int(
        stage3["x_m"].isna().sum()
    )

    stage3_nan_y = int(
        stage3["y_m"].isna().sum()
    )

    print(
        f"  Stage 3 GNSS x NaN samples: {stage3_nan_x}"
    )

    print(
        f"  Stage 3 GNSS y NaN samples: {stage3_nan_y}"
    )

    if stage3_nan_x == 300 and stage3_nan_y == 300:

        print(
            "  Stage 3 GNSS outage representation: PASS"
        )

    else:

        print(
            "  Stage 3 GNSS outage representation: CHECK"
        )


# ============================================================
# STEP 4: COMPUTE TIME STEP
# ============================================================

def compute_dt(df):

    timestamps = df[
        "timestamp"
    ].to_numpy(dtype=float)

    if not np.isfinite(timestamps).all():
        raise ValueError(
            "Timestamp contains NaN or infinite values."
        )

    dt = np.diff(
        timestamps,
        prepend=timestamps[0]
    )

    dt[0] = 0.0

    valid = dt[
        (dt > 0)
        &
        np.isfinite(dt)
    ]

    if len(valid) == 0:
        raise ValueError(
            "Could not determine a valid time step."
        )

    fallback = float(
        np.median(valid)
    )

    bad = (
        ~np.isfinite(dt)
        |
        (dt <= 0)
    )

    bad[0] = False

    dt[bad] = fallback

    df["dt_s"] = dt

    print("\nTime-step calculation: PASS")

    print(
        f"  Median dt: "
        f"{np.median(dt[1:]):.3f} s"
    )

    return df


# ============================================================
# STEP 5: COMPUTE REFERENCE HEADING
# ============================================================

def compute_heading(df):

    heading_rad = df[
        "ground_heading"
    ].to_numpy(dtype=float)

    if not np.isfinite(heading_rad).all():
        raise ValueError(
            "ground_heading contains invalid values."
        )

    heading_deg = (
        np.degrees(heading_rad)
        % 360.0
    )

    df["heading_deg"] = heading_deg

    print(
        "\nStable vehicle heading obtained from Stage 2 "
        "reference trajectory: PASS"
    )

    print(
        f"  Heading range: "
        f"{heading_deg.min():.2f}° to "
        f"{heading_deg.max():.2f}°"
    )

    return df


# ============================================================
# STEP 6: BUILD CONTINUOUS GROUND TRUTH
# ============================================================

def build_ground_truth(df):

    """
    Build a continuous reference trajectory from the known
    simulated vehicle speed and heading.

    This is NOT GNSS.

    It represents the true trajectory used only for evaluation.

    GNSS availability therefore does not create NaN values in
    the ground truth trajectory.
    """

    n = len(df)

    gt_x = np.zeros(n)
    gt_y = np.zeros(n)

    speed = df[
        "ground_speed"
    ].to_numpy(dtype=float)

    heading_rad = np.radians(
        df[
            "heading_deg"
        ].to_numpy(dtype=float)
    )

    dt = df[
        "dt_s"
    ].to_numpy(dtype=float)

    if not np.isfinite(speed).all():
        raise ValueError(
            "ground_speed contains invalid values."
        )

    for i in range(1, n):

        dx = (
            speed[i]
            *
            np.cos(heading_rad[i])
            *
            dt[i]
        )

        dy = (
            speed[i]
            *
            np.sin(heading_rad[i])
            *
            dt[i]
        )

        gt_x[i] = (
            gt_x[i - 1]
            +
            dx
        )

        gt_y[i] = (
            gt_y[i - 1]
            +
            dy
        )

    gt_x[0] = 0.0
    gt_y[0] = 0.0

    df["ground_truth_x_m"] = gt_x
    df["ground_truth_y_m"] = gt_y

    print(
        "\nContinuous ground truth trajectory: PASS"
    )

    print(
        f"  Ground truth samples: {n}"
    )

    print(
        f"  Ground truth x range: "
        f"{gt_x.min():.2f} to {gt_x.max():.2f} m"
    )

    print(
        f"  Ground truth y range: "
        f"{gt_y.min():.2f} to {gt_y.max():.2f} m"
    )

    return df


# ============================================================
# STEP 7: CREATE IMPERFECT DR SENSOR ESTIMATES
# ============================================================

def build_dr_estimates(df):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n = len(df)

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speed_noise = rng.normal(
        0.0,
        SPEED_NOISE_STD_MPS,
        n
    )

    ground_speed = df[
        "ground_speed"
    ].to_numpy(dtype=float)

    dr_speed = (
        ground_speed
        +
        SPEED_BIAS_MPS
        +
        speed_noise
    )

    dr_speed = np.clip(
        dr_speed,
        0.0,
        None
    )

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    heading_noise = rng.normal(
        0.0,
        HEADING_NOISE_STD_DEG,
        n
    )

    dr_heading = (
        df["heading_deg"].to_numpy(dtype=float)
        +
        HEADING_BIAS_DEG
        +
        heading_noise
    )

    dr_heading = (
        dr_heading
        % 360.0
    )

    df["dr_speed_mps"] = dr_speed
    df["dr_heading_deg"] = dr_heading

    print(
        "\nControlled DR errors added: PASS"
    )

    print(
        f"  Speed bias: "
        f"{SPEED_BIAS_MPS:.3f} m/s"
    )

    print(
        f"  Heading bias: "
        f"{HEADING_BIAS_DEG:.3f}°"
    )

    return df


# ============================================================
# STEP 8: DEAD RECKONING INTEGRATION
# ============================================================

def run_dead_reckoning(df):

    n = len(df)

    dr_x = np.zeros(n)
    dr_y = np.zeros(n)

    speed = df[
        "dr_speed_mps"
    ].to_numpy(dtype=float)

    heading_rad = np.radians(
        df[
            "dr_heading_deg"
        ].to_numpy(dtype=float)
    )

    dt = df[
        "dt_s"
    ].to_numpy(dtype=float)

    for i in range(1, n):

        dx = (
            speed[i]
            *
            np.cos(heading_rad[i])
            *
            dt[i]
        )

        dy = (
            speed[i]
            *
            np.sin(heading_rad[i])
            *
            dt[i]
        )

        dr_x[i] = (
            dr_x[i - 1]
            +
            dx
        )

        dr_y[i] = (
            dr_y[i - 1]
            +
            dy
        )

    dr_x[0] = 0.0
    dr_y[0] = 0.0

    df["dr_x_m"] = dr_x
    df["dr_y_m"] = dr_y

    print(
        "\nDead Reckoning integration: PASS"
    )

    return df


# ============================================================
# STEP 9: POSITION ERROR
# ============================================================

def compute_error(df):

    dx = (
        df["ground_truth_x_m"]
        -
        df["dr_x_m"]
    )

    dy = (
        df["ground_truth_y_m"]
        -
        df["dr_y_m"]
    )

    error = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    if not np.isfinite(error).all():
        raise ValueError(
            "Position error contains NaN or infinite values."
        )

    df["position_error_m"] = error

    return df


# ============================================================
# STEP 10: OUTAGE ANALYSIS
# ============================================================

def analyze_outage(df):

    before = df[
        df["timestamp"] < OUTAGE_START
    ]

    outage = df[
        (df["timestamp"] >= OUTAGE_START)
        &
        (df["timestamp"] < OUTAGE_END)
    ]

    after = df[
        df["timestamp"] >= OUTAGE_END
    ]

    print("\nGNSS outage analysis:")

    # --------------------------------------------------------
    # GNSS availability
    # --------------------------------------------------------

    outage_gps = outage[
        "gps_available"
    ].to_numpy()

    if np.all(outage_gps == 0):

        print(
            "  GNSS unavailable during 60–90 s: PASS"
        )

    else:

        print(
            "  GNSS availability during outage: CHECK"
        )

    # --------------------------------------------------------
    # Ground truth availability
    # --------------------------------------------------------

    gt_valid = (
        outage["ground_truth_x_m"].notna()
        &
        outage["ground_truth_y_m"].notna()
    )

    if gt_valid.all():

        print(
            "  Ground truth available during outage: PASS"
        )

    else:

        print(
            "  Ground truth available during outage: FAIL"
        )

    # --------------------------------------------------------
    # DR availability
    # --------------------------------------------------------

    dr_valid = (
        outage["dr_x_m"].notna()
        &
        outage["dr_y_m"].notna()
    )

    if dr_valid.all():

        print(
            "  DR available during outage: PASS"
        )

    else:

        print(
            "  DR available during outage: FAIL"
        )

    # --------------------------------------------------------
    # Error before outage
    # --------------------------------------------------------

    if len(before) > 0:

        error_before = float(
            before["position_error_m"].iloc[-1]
        )

        print(
            f"  Error at outage start: "
            f"{error_before:.2f} m"
        )

    # --------------------------------------------------------
    # Outage error
    # --------------------------------------------------------

    if len(outage) > 0:

        outage_errors = outage[
            "position_error_m"
        ].to_numpy(dtype=float)

        valid_errors = outage_errors[
            np.isfinite(outage_errors)
        ]

        if len(valid_errors) > 0:

            maximum_error = float(
                np.max(valid_errors)
            )

            end_error = float(
                valid_errors[-1]
            )

            print(
                f"  Maximum error during outage: "
                f"{maximum_error:.2f} m"
            )

            print(
                f"  Error at end of outage: "
                f"{end_error:.2f} m"
            )

    # --------------------------------------------------------
    # Final error
    # --------------------------------------------------------

    if len(after) > 0:

        final_error = float(
            after["position_error_m"].iloc[-1]
        )

        print(
            f"  Final error: "
            f"{final_error:.2f} m"
        )


# ============================================================
# STEP 11: VALIDATION
# ============================================================

def validate(df):

    print()
    print("=" * 70)
    print("STAGE 4 VALIDATION")
    print("=" * 70)

    passed = True

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(df) == 1200:

        print(
            "[PASS] Row count = 1200"
        )

    else:

        print(
            f"[FAIL] Row count = {len(df)}"
        )

        passed = False

    # --------------------------------------------------------
    # DR starts at zero
    # --------------------------------------------------------

    if (
        abs(df["dr_x_m"].iloc[0]) < 1e-9
        and
        abs(df["dr_y_m"].iloc[0]) < 1e-9
    ):

        print(
            "[PASS] DR starts at (0, 0)"
        )

    else:

        print(
            "[FAIL] DR does not start at (0, 0)"
        )

        passed = False

    # --------------------------------------------------------
    # Ground truth starts at zero
    # --------------------------------------------------------

    if (
        abs(df["ground_truth_x_m"].iloc[0]) < 1e-9
        and
        abs(df["ground_truth_y_m"].iloc[0]) < 1e-9
    ):

        print(
            "[PASS] Ground truth starts at (0, 0)"
        )

    else:

        print(
            "[FAIL] Ground truth does not start at (0, 0)"
        )

        passed = False

    # --------------------------------------------------------
    # GPS availability
    # --------------------------------------------------------

    gps_available = int(
        df["gps_available"].sum()
    )

    gps_outage = int(
        (df["gps_available"] == 0).sum()
    )

    print(
        f"GPS available samples: "
        f"{gps_available}"
    )

    print(
        f"GPS outage samples: "
        f"{gps_outage}"
    )

    if gps_available == 900:

        print(
            "[PASS] GPS available samples = 900"
        )

    else:

        print(
            "[FAIL] Unexpected GPS availability"
        )

        passed = False

    if gps_outage == 300:

        print(
            "[PASS] GPS outage samples = 300"
        )

    else:

        print(
            "[FAIL] Unexpected GPS outage length"
        )

        passed = False

    # --------------------------------------------------------
    # Exact outage interval
    # --------------------------------------------------------

    outage = df[
        (df["timestamp"] >= OUTAGE_START)
        &
        (df["timestamp"] < OUTAGE_END)
    ]

    if len(outage) == 300:

        print(
            "[PASS] GNSS outage interval = 60–90 s"
        )

    else:

        print(
            f"[FAIL] Expected 300 outage samples, "
            f"found {len(outage)}"
        )

        passed = False

    # --------------------------------------------------------
    # GPS must be unavailable during outage
    # --------------------------------------------------------

    if (
        len(outage) == 300
        and
        (outage["gps_available"] == 0).all()
    ):

        print(
            "[PASS] GNSS unavailable during outage"
        )

    else:

        print(
            "[FAIL] GNSS outage representation"
        )

        passed = False

    # --------------------------------------------------------
    # Ground truth must remain available
    # --------------------------------------------------------

    if (
        df["ground_truth_x_m"].notna().all()
        and
        df["ground_truth_y_m"].notna().all()
    ):

        print(
            "[PASS] Ground truth available for all 1200 samples"
        )

    else:

        print(
            "[FAIL] Ground truth contains NaN values"
        )

        passed = False

    # --------------------------------------------------------
    # DR must continue through outage
    # --------------------------------------------------------

    if (
        len(outage) == 300
        and
        outage["dr_x_m"].notna().all()
        and
        outage["dr_y_m"].notna().all()
    ):

        print(
            "[PASS] DR continues through 60–90 s outage"
        )

    else:

        print(
            "[FAIL] DR outage handling"
        )

        passed = False

    # --------------------------------------------------------
    # Position error validity
    # --------------------------------------------------------

    if (
        df["position_error_m"].notna().all()
        and
        np.isfinite(
            df["position_error_m"]
        ).all()
    ):

        print(
            "[PASS] Position error contains no NaN values"
        )

    else:

        print(
            "[FAIL] Position error contains NaN/inf values"
        )

        passed = False

    # --------------------------------------------------------
    # Initial error
    # --------------------------------------------------------

    initial_error = float(
        df["position_error_m"].iloc[0]
    )

    if initial_error < 1e-9:

        print(
            "[PASS] Initial position error = 0 m"
        )

    else:

        print(
            f"[FAIL] Initial error = "
            f"{initial_error:.3f} m"
        )

        passed = False

    # --------------------------------------------------------
    # Final error
    # --------------------------------------------------------

    final_error = float(
        df["position_error_m"].iloc[-1]
    )

    if final_error > 0.0:

        print(
            f"[PASS] Final DR error = "
            f"{final_error:.2f} m"
        )

    else:

        print(
            "[FAIL] No DR drift detected"
        )

        passed = False

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_output = [
        "timestamp",
        "gps_available",
        "ground_truth_x_m",
        "ground_truth_y_m",
        "heading_deg",
        "dt_s",
        "dr_speed_mps",
        "dr_heading_deg",
        "dr_x_m",
        "dr_y_m",
        "position_error_m"
    ]

    missing = [
        c
        for c in required_output
        if c not in df.columns
    ]

    if not missing:

        print(
            "[PASS] Required output columns exist"
        )

    else:

        print(
            f"[FAIL] Missing output columns: "
            f"{missing}"
        )

        passed = False

    # --------------------------------------------------------
    # Outage error increase
    # --------------------------------------------------------

    outage_errors = outage[
        "position_error_m"
    ].to_numpy(dtype=float)

    valid_outage_errors = outage_errors[
        np.isfinite(outage_errors)
    ]

    if len(valid_outage_errors) > 1:

        first_error = float(
            valid_outage_errors[0]
        )

        last_error = float(
            valid_outage_errors[-1]
        )

        maximum_error = float(
            np.max(valid_outage_errors)
        )

        print(
            f"  Error at outage start: "
            f"{first_error:.2f} m"
        )

        print(
            f"  Maximum outage error: "
            f"{maximum_error:.2f} m"
        )

        print(
            f"  Error at outage end: "
            f"{last_error:.2f} m"
        )

        if last_error > first_error:

            print(
                "[PASS] DR error increases during GNSS outage"
            )

        else:

            print(
                "[WARNING] DR error did not increase during outage"
            )

    else:

        print(
            "[FAIL] Insufficient valid outage errors"
        )

        passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    if passed:

        print(
            "Stage 4 validation: PASSED"
        )

    else:

        print(
            "Stage 4 validation: FAILED"
        )

    return passed


# ============================================================
# STEP 12: PLOT
# ============================================================

def plot_trajectories(df):

    output_directory = os.path.dirname(
        OUTPUT_PLOT
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # Ground truth
    ax.plot(
        df["ground_truth_x_m"],
        df["ground_truth_y_m"],
        label="Ground Truth",
        linewidth=1.8
    )

    # Dead Reckoning
    ax.plot(
        df["dr_x_m"],
        df["dr_y_m"],
        label="Dead Reckoning",
        linestyle="--",
        linewidth=1.5
    )

    # Start
    ax.scatter(
        df["ground_truth_x_m"].iloc[0],
        df["ground_truth_y_m"].iloc[0],
        label="Start",
        zorder=5
    )

    # Ground truth end
    ax.scatter(
        df["ground_truth_x_m"].iloc[-1],
        df["ground_truth_y_m"].iloc[-1],
        marker="x",
        s=70,
        label="Ground Truth End",
        zorder=5
    )

    # DR end
    ax.scatter(
        df["dr_x_m"].iloc[-1],
        df["dr_y_m"].iloc[-1],
        marker="x",
        s=70,
        label="DR End",
        zorder=5
    )

    ax.set_xlabel(
        "X Position (m)"
    )

    ax.set_ylabel(
        "Y Position (m)"
    )

    ax.set_title(
        "Stage 4: Ground Truth vs Dead Reckoning"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend()

    ax.set_aspect(
        "equal",
        adjustable="datalim"
    )

    fig.tight_layout()

    os.makedirs(
        os.path.dirname(OUTPUT_PLOT),
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_PLOT,
        dpi=150
    )

    plt.close(fig)

    print(
        f"\nPlot saved to:\n"
        f"  {OUTPUT_PLOT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Load datasets
    stage2 = load_stage2()
    stage3 = load_stage3()

    # Verify Stage 2 and Stage 3
    verify_stage_data(
        stage2,
        stage3
    )

    # Work primarily from Stage 2
    df = stage2.copy()

    # Time step
    df = compute_dt(
        df
    )

    # Reference heading
    df = compute_heading(
        df
    )

    # Continuous ground truth
    df = build_ground_truth(
        df
    )

    # Imperfect DR measurements
    df = build_dr_estimates(
        df
    )

    # Dead Reckoning
    df = run_dead_reckoning(
        df
    )

    # Position error
    df = compute_error(
        df
    )

    # Outage analysis
    analyze_outage(
        df
    )

    # Save CSV
    output_directory = os.path.dirname(
        OUTPUT_CSV
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nDead Reckoning dataset saved to:\n"
        f"  {OUTPUT_CSV}"
    )

    # Plot
    plot_trajectories(
        df
    )

    # Validation
    validate(
        df
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()