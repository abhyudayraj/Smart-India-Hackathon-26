"""
STAGE 6 — Dead Reckoning Drift Analysis

Input:
    stage5/data/gnss_outage.csv

Output:
    stage6/data/drift_analysis.csv
    stage6/outputs/stage6_error_over_time.png
    stage6/outputs/stage6_outage_trajectory.png
    stage6/outputs/stage6_drift_vs_distance.png

Purpose:
    Measure Dead Reckoning position error against ground truth,
    with specific analysis during the simulated GNSS outage.

Important:
    This stage only measures drift.
    It does NOT correct the Dead Reckoning trajectory.
    dr_x_m and dr_y_m are never modified.
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

INPUT_PATH = "stage5/data/gnss_outage.csv"

OUTPUT_CSV_PATH = "stage6/data/drift_analysis.csv"

OUTPUT_ERROR_OVER_TIME_PLOT = (
    "stage6/outputs/stage6_error_over_time.png"
)

OUTPUT_OUTAGE_TRAJECTORY_PLOT = (
    "stage6/outputs/stage6_outage_trajectory.png"
)

OUTPUT_DRIFT_VS_DISTANCE_PLOT = (
    "stage6/outputs/stage6_drift_vs_distance.png"
)


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "ground_truth_x_m",
    "ground_truth_y_m",
    "dr_x_m",
    "dr_y_m",
    "gnss_available",
    "cumulative_distance_m",
]


# ============================================================
# LOAD INPUT
# ============================================================

def load_input(path):

    if not os.path.exists(path):

        print(f"ERROR: Input file not found: {path}")
        print()
        print(
            "Make sure Stage 5 has been run and the file exists at:"
        )
        print("stage5/data/gnss_outage.csv")

        sys.exit(1)

    df = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        print(
            "ERROR: Input file is missing required column(s):"
        )

        for column in missing:
            print(f"  - {column}")

        sys.exit(1)

    return df


# ============================================================
# CALCULATE DEAD RECKONING ERROR
# ============================================================

def calculate_dr_error(df):

    df = df.copy()

    # Error = Ground Truth - Dead Reckoning
    df["error_x_m"] = (
        df["ground_truth_x_m"] - df["dr_x_m"]
    )

    df["error_y_m"] = (
        df["ground_truth_y_m"] - df["dr_y_m"]
    )

    # Euclidean position error
    df["dr_error_m"] = np.sqrt(
        df["error_x_m"] ** 2
        + df["error_y_m"] ** 2
    )

    return df


# ============================================================
# DISTANCE SINCE OUTAGE START
# ============================================================

def calculate_distance_since_outage(
    df,
    outage_mask,
    outage_start_distance,
):

    df = df.copy()

    distance_since_outage_m = np.full(
        len(df),
        np.nan,
    )

    if outage_mask.any():

        distance_since_outage_m = np.where(
            outage_mask,
            df["cumulative_distance_m"]
            - outage_start_distance,
            np.nan,
        )

    df["distance_since_outage_m"] = (
        distance_since_outage_m
    )

    return df


# ============================================================
# ERROR VS TIME PLOT
# ============================================================

def make_error_over_time_plot(
    df,
    outage_mask,
    path,
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x_axis = df["timestamp"]

    # DR position error
    ax.plot(
        x_axis,
        df["dr_error_m"],
        color="tab:red",
        linewidth=1.5,
        label="DR Position Error",
    )

    # Highlight GNSS outage
    if outage_mask.any():

        outage_x = x_axis[outage_mask]

        ax.axvspan(
            outage_x.iloc[0],
            outage_x.iloc[-1],
            color="orange",
            alpha=0.2,
            label="GNSS Outage",
        )

    ax.set_title(
        "Dead Reckoning Drift During GNSS Outage"
    )

    ax.set_xlabel("Time (s)")

    ax.set_ylabel(
        "Dead Reckoning Position Error (m)"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# OUTAGE TRAJECTORY PLOT
# ============================================================

def make_outage_trajectory_plot(
    df,
    outage_mask,
    path,
):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    if outage_mask.any():

        outage_df = df.loc[outage_mask]

        # Ground Truth
        ax.plot(
            outage_df["ground_truth_x_m"],
            outage_df["ground_truth_y_m"],
            color="tab:green",
            linewidth=2,
            label="Ground Truth",
        )

        # Dead Reckoning
        ax.plot(
            outage_df["dr_x_m"],
            outage_df["dr_y_m"],
            color="tab:red",
            linewidth=2,
            linestyle="--",
            label="Dead Reckoning",
        )

        first_idx = outage_df.index[0]
        last_idx = outage_df.index[-1]

        # Outage start
        ax.scatter(
            df.loc[
                first_idx,
                "ground_truth_x_m",
            ],
            df.loc[
                first_idx,
                "ground_truth_y_m",
            ],
            color="black",
            marker="x",
            s=100,
            zorder=4,
            label="Outage Start",
        )

        # Outage end
        ax.scatter(
            df.loc[
                last_idx,
                "ground_truth_x_m",
            ],
            df.loc[
                last_idx,
                "ground_truth_y_m",
            ],
            color="purple",
            marker="x",
            s=100,
            zorder=4,
            label="Outage End",
        )

        ax.set_aspect(
            "equal",
            adjustable="datalim",
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No GNSS outage samples found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(
        "Ground Truth vs Dead Reckoning During GNSS Outage"
    )

    ax.set_xlabel(
        "X Position (m)"
    )

    ax.set_ylabel(
        "Y Position (m)"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# DRIFT VS DISTANCE PLOT
# ============================================================

def make_drift_vs_distance_plot(
    df,
    outage_mask,
    path,
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    if outage_mask.any():

        outage_df = df.loc[outage_mask]

        ax.plot(
            outage_df["distance_since_outage_m"],
            outage_df["dr_error_m"],
            color="tab:red",
            linewidth=2,
            marker="o",
            markersize=3,
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No GNSS outage samples found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(
        "Dead Reckoning Error Growth During GNSS Outage"
    )

    ax.set_xlabel(
        "Distance Since Outage Start (m)"
    )

    ax.set_ylabel(
        "Dead Reckoning Position Error (m)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

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

def validate_stage6(
    df,
    outage_mask,
):

    print()
    print("====================================================")
    print("STAGE 6 VALIDATION")
    print("====================================================")

    # --------------------------------------------------------
    # Check error columns
    # --------------------------------------------------------

    required_output_columns = [
        "error_x_m",
        "error_y_m",
        "dr_error_m",
        "distance_since_outage_m",
    ]

    missing = [
        column
        for column in required_output_columns
        if column not in df.columns
    ]

    if missing:

        print("FAIL: Missing calculated columns:")

        for column in missing:
            print(f"  - {column}")

        return False

    print("Calculated error columns: PASS")

    # --------------------------------------------------------
    # Check error calculation
    # --------------------------------------------------------

    expected_error = np.sqrt(
        (
            df["ground_truth_x_m"]
            - df["dr_x_m"]
        ) ** 2
        +
        (
            df["ground_truth_y_m"]
            - df["dr_y_m"]
        ) ** 2
    )

    error_match = np.allclose(
        df["dr_error_m"].to_numpy(),
        expected_error.to_numpy(),
        atol=1e-9,
        equal_nan=False,
    )

    if not error_match:

        print(
            "FAIL: DR error calculation is incorrect."
        )

        return False

    print("DR error calculation: PASS")

    # --------------------------------------------------------
    # Check outage exists
    # --------------------------------------------------------

    if not outage_mask.any():

        print(
            "FAIL: No GNSS outage samples found."
        )

        return False

    print(
        f"GNSS outage samples: "
        f"{int(outage_mask.sum())}"
    )

    # --------------------------------------------------------
    # Check distance since outage
    # --------------------------------------------------------

    outage_df = df.loc[outage_mask]

    distance_values = (
        outage_df["distance_since_outage_m"]
    )

    if distance_values.isna().any():

        print(
            "FAIL: Distance since outage contains NaN "
            "during outage."
        )

        return False

    first_distance = distance_values.iloc[0]

    if abs(first_distance) > 1e-9:

        print(
            "FAIL: Distance since outage does not "
            "start at zero."
        )

        return False

    if not np.all(
        np.diff(distance_values.to_numpy()) >= 0
    ):

        print(
            "FAIL: Distance since outage is not "
            "monotonically increasing."
        )

        return False

    print(
        "Distance since outage calculation: PASS"
    )

    # --------------------------------------------------------
    # Check DR data was not modified
    # --------------------------------------------------------

    if df["dr_x_m"].isna().any():
        print(
            "WARNING: dr_x_m contains NaN values."
        )

    if df["dr_y_m"].isna().any():
        print(
            "WARNING: dr_y_m contains NaN values."
        )

    print(
        "Dead Reckoning trajectory preserved: PASS"
    )

    # --------------------------------------------------------
    # Check error values
    # --------------------------------------------------------

    if df["dr_error_m"].isna().any():

        print(
            "FAIL: DR error contains NaN values."
        )

        return False

    if (df["dr_error_m"] < 0).any():

        print(
            "FAIL: Negative DR error detected."
        )

        return False

    print(
        "DR error values: PASS"
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print()
    print("Stage 6 validation: PASS")

    return True


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    df,
    outage_mask,
    outage_stats,
):

    total_distance = (
        df["cumulative_distance_m"].iloc[-1]
    )

    print()
    print("====================================================")
    print("STAGE 6: DEAD RECKONING DRIFT ANALYSIS")
    print("====================================================")

    print(
        f"Total Route Distance: "
        f"{total_distance:.2f} m"
    )

    print()

    print("GNSS Outage:")

    if outage_mask.any():

        print(
            f"Start Distance: "
            f"{outage_stats['start_distance']:.2f} m"
        )

        print(
            f"End Distance: "
            f"{outage_stats['end_distance']:.2f} m"
        )

        print(
            f"Outage Distance: "
            f"{outage_stats['outage_distance']:.2f} m"
        )

        print(
            f"Outage Samples: "
            f"{outage_stats['outage_samples']}"
        )

        if outage_stats["outage_duration"] is not None:

            print(
                f"Outage Duration: "
                f"{outage_stats['outage_duration']:.2f} s"
            )

        print()

        print(
            "DEAD RECKONING ERROR DURING OUTAGE:"
        )

        print()

        print(
            f"Mean Error: "
            f"{outage_stats['mean_error']:.3f} m"
        )

        print(
            f"Median Error: "
            f"{outage_stats['median_error']:.3f} m"
        )

        print(
            f"Minimum Error: "
            f"{outage_stats['min_error']:.3f} m"
        )

        print(
            f"Maximum Error: "
            f"{outage_stats['max_error']:.3f} m"
        )

        print(
            f"Final Error: "
            f"{outage_stats['final_error']:.3f} m"
        )

        print(
            f"RMSE: "
            f"{outage_stats['rmse']:.3f} m"
        )

    else:

        print(
            "No GNSS outage samples found "
            "in this dataset."
        )

    print()

    print("Stage 6 Output:")
    print(
        f"{OUTPUT_CSV_PATH}"
    )

    print()

    print("Generated Plots:")
    print(
        f"{OUTPUT_ERROR_OVER_TIME_PLOT}"
    )
    print(
        f"{OUTPUT_OUTAGE_TRAJECTORY_PLOT}"
    )
    print(
        f"{OUTPUT_DRIFT_VS_DISTANCE_PLOT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Create Stage 6 directories
    os.makedirs(
        "stage6/data",
        exist_ok=True,
    )

    os.makedirs(
        "stage6/outputs",
        exist_ok=True,
    )

    # Load Stage 5 dataset
    df = load_input(INPUT_PATH)

    print("Loaded Stage 5 dataset:")
    print(
        f"  {INPUT_PATH}"
    )
    print(
        f"  Rows: {len(df)}"
    )
    print(
        f"  Columns: {len(df.columns)}"
    )

    # Calculate DR error
    df = calculate_dr_error(df)

    # Identify simulated GNSS outage
    outage_mask = (
        df["gnss_available"] == False
    )  # noqa: E712

    outage_stats = {}

    outage_start_distance = 0.0

    # --------------------------------------------------------
    # Calculate outage statistics
    # --------------------------------------------------------

    if outage_mask.any():

        outage_df = df.loc[outage_mask]

        outage_start_distance = (
            outage_df[
                "cumulative_distance_m"
            ].iloc[0]
        )

        outage_end_distance = (
            outage_df[
                "cumulative_distance_m"
            ].iloc[-1]
        )

        outage_duration = None

        if "timestamp" in df.columns:

            try:

                outage_duration = float(
                    outage_df[
                        "timestamp"
                    ].iloc[-1]
                    -
                    outage_df[
                        "timestamp"
                    ].iloc[0]
                )

            except (
                TypeError,
                ValueError,
            ):

                outage_duration = None

        outage_errors = (
            outage_df["dr_error_m"]
        )

        outage_stats = {

            "start_distance":
                outage_start_distance,

            "end_distance":
                outage_end_distance,

            "outage_distance":
                outage_end_distance
                - outage_start_distance,

            "outage_samples":
                int(outage_mask.sum()),

            "outage_duration":
                outage_duration,

            "mean_error":
                outage_errors.mean(),

            "median_error":
                outage_errors.median(),

            "min_error":
                outage_errors.min(),

            "max_error":
                outage_errors.max(),

            "final_error":
                outage_errors.iloc[-1],

            "rmse":
                np.sqrt(
                    np.mean(
                        outage_errors ** 2
                    )
                ),
        }

    # --------------------------------------------------------
    # Calculate distance since outage
    # --------------------------------------------------------

    df = calculate_distance_since_outage(
        df,
        outage_mask,
        outage_start_distance,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    make_error_over_time_plot(
        df,
        outage_mask,
        OUTPUT_ERROR_OVER_TIME_PLOT,
    )

    make_outage_trajectory_plot(
        df,
        outage_mask,
        OUTPUT_OUTAGE_TRAJECTORY_PLOT,
    )

    make_drift_vs_distance_plot(
        df,
        outage_mask,
        OUTPUT_DRIFT_VS_DISTANCE_PLOT,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_passed = validate_stage6(
        df,
        outage_mask,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        df,
        outage_mask,
        outage_stats,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    if not validation_passed:

        print("STAGE 6 RESULT: FAIL")

        sys.exit(1)

    print("STAGE 6 RESULT: PASS")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()