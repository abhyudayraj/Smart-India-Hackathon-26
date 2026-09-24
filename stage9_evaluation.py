"""
STAGE 9 — REAL DR vs AI-CORRECTED EVALUATION

Input:
    stage8/data/ml_corrected_trajectory.csv

Uses:
    x_true, y_true       -> Ground Truth
    x_dr, y_dr           -> Raw Dead Reckoning
    x_ai, y_ai           -> AI Corrected

Output:
    stage9/data/error_timeseries.csv
    stage9/data/evaluation_metrics.json

Plots:
    stage9/outputs/stage9_trajectory_comparison.png
    stage9/outputs/stage9_error_comparison.png
    stage9/outputs/stage9_outage_error.png

IMPORTANT:
    This stage uses REAL pipeline data from Stage 8.
    No synthetic data is generated.
"""


from __future__ import annotations

import json
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

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "stage8",
    "data",
    "ml_corrected_trajectory.csv",
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "stage9",
    "data",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "stage9",
    "outputs",
)

ERROR_TIMESERIES_PATH = os.path.join(
    DATA_DIR,
    "error_timeseries.csv",
)

METRICS_PATH = os.path.join(
    DATA_DIR,
    "evaluation_metrics.json",
)

TRAJECTORY_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage9_trajectory_comparison.png",
)

ERROR_COMPARISON_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage9_error_comparison.png",
)

OUTAGE_ERROR_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage9_outage_error.png",
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "x_true",
    "y_true",
    "x_dr",
    "y_dr",
    "x_ai",
    "y_ai",
    "gnss_available",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_input():
    if not os.path.exists(INPUT_PATH):
        print("ERROR: Stage 8 output not found:")
        print(f"  {INPUT_PATH}")
        print()
        print("Make sure Stage 8 has been run successfully.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        print("ERROR: Stage 8 dataset is missing required columns:")

        for column in missing:
            print(f"  - {column}")

        sys.exit(1)

    return df


# ============================================================
# CALCULATE POSITION ERRORS
# ============================================================

def calculate_errors(df):
    df = df.copy()

    # Raw DR error
    df["dr_error_x"] = (
        df["x_dr"] - df["x_true"]
    )

    df["dr_error_y"] = (
        df["y_dr"] - df["y_true"]
    )

    df["dr_error_m"] = np.sqrt(
        df["dr_error_x"] ** 2
        + df["dr_error_y"] ** 2
    )

    # AI corrected error
    df["ai_error_x"] = (
        df["x_ai"] - df["x_true"]
    )

    df["ai_error_y"] = (
        df["y_ai"] - df["y_true"]
    )

    df["ai_error_m"] = np.sqrt(
        df["ai_error_x"] ** 2
        + df["ai_error_y"] ** 2
    )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(error):
    error = np.asarray(error, dtype=float)

    error = error[np.isfinite(error)]

    if len(error) == 0:
        return {
            "MAE_m": None,
            "RMSE_m": None,
            "max_error_m": None,
            "final_error_m": None,
            "median_error_m": None,
            "p95_error_m": None,
        }

    return {
        "MAE_m": float(np.mean(np.abs(error))),

        "RMSE_m": float(
            np.sqrt(np.mean(error ** 2))
        ),

        "max_error_m": float(
            np.max(np.abs(error))
        ),

        "final_error_m": float(
            np.abs(error[-1])
        ),

        "median_error_m": float(
            np.median(np.abs(error))
        ),

        "p95_error_m": float(
            np.percentile(np.abs(error), 95)
        ),
    }


# ============================================================
# IMPROVEMENT
# ============================================================

def calculate_improvement(dr_metric, ai_metric):
    if dr_metric == 0:
        return 0.0

    return (
        (dr_metric - ai_metric)
        / dr_metric
        * 100.0
    )


# ============================================================
# TRAJECTORY PLOT
# ============================================================

def make_trajectory_plot(df):

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        df["x_true"],
        df["y_true"],
        linewidth=2,
        label="Ground Truth",
    )

    ax.plot(
        df["x_dr"],
        df["y_dr"],
        linestyle="--",
        linewidth=1.8,
        label="Raw Dead Reckoning",
    )

    ax.plot(
        df["x_ai"],
        df["y_ai"],
        linestyle="-.",
        linewidth=1.8,
        label="AI Corrected",
    )

    # Mark outage samples
    outage_mask = (
        df["gnss_available"] == False
    )

    if outage_mask.any():

        outage_df = df.loc[outage_mask]

        ax.scatter(
            outage_df["x_true"],
            outage_df["y_true"],
            s=10,
            alpha=0.35,
            label="GNSS Outage",
        )

    ax.set_title(
        "Stage 9: Ground Truth vs DR vs AI Corrected"
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    fig.tight_layout()

    fig.savefig(
        TRAJECTORY_PLOT_PATH,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# ERROR COMPARISON PLOT
# ============================================================

def make_error_comparison_plot(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["timestamp"],
        df["dr_error_m"],
        linewidth=1.5,
        label="Raw DR Error",
    )

    ax.plot(
        df["timestamp"],
        df["ai_error_m"],
        linewidth=1.5,
        label="AI Corrected Error",
    )

    outage_mask = (
        df["gnss_available"] == False
    )

    if outage_mask.any():

        outage_time = df.loc[
            outage_mask,
            "timestamp"
        ]

        ax.axvspan(
            outage_time.iloc[0],
            outage_time.iloc[-1],
            alpha=0.2,
            label="GNSS Outage",
        )

    ax.set_title(
        "Stage 9: Position Error Comparison"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        ERROR_COMPARISON_PLOT_PATH,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# OUTAGE ERROR PLOT
# ============================================================

def make_outage_error_plot(df):

    outage_mask = (
        df["gnss_available"] == False
    )

    if not outage_mask.any():
        return

    outage_df = df.loc[outage_mask].copy()

    # Distance from first outage sample
    if "distance_since_outage_m" in outage_df.columns:

        x_axis = outage_df[
            "distance_since_outage_m"
        ]

        x_label = "Distance Since Outage Start (m)"

    else:

        x_axis = (
            outage_df["timestamp"]
            - outage_df["timestamp"].iloc[0]
        )

        x_label = "Time Since Outage Start (s)"

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x_axis,
        outage_df["dr_error_m"],
        linewidth=2,
        label="Raw DR Error",
    )

    ax.plot(
        x_axis,
        outage_df["ai_error_m"],
        linewidth=2,
        label="AI Corrected Error",
    )

    ax.set_title(
        "Stage 9: Error During GNSS Outage"
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Position Error (m)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTAGE_ERROR_PLOT_PATH,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# SAVE ERROR TIMESERIES
# ============================================================

def save_error_timeseries(df):

    columns = [
        "timestamp",
        "gnss_available",
        "dr_error_x",
        "dr_error_y",
        "dr_error_m",
        "ai_error_x",
        "ai_error_y",
        "ai_error_m",
    ]

    if "distance_since_outage_m" in df.columns:
        columns.append(
            "distance_since_outage_m"
        )

    output = df[columns].copy()

    output.to_csv(
        ERROR_TIMESERIES_PATH,
        index=False,
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    df,
    dr_metrics,
    ai_metrics,
    outage_dr_metrics,
    outage_ai_metrics,
):

    overall_rmse_improvement = (
        calculate_improvement(
            dr_metrics["RMSE_m"],
            ai_metrics["RMSE_m"],
        )
    )

    overall_mae_improvement = (
        calculate_improvement(
            dr_metrics["MAE_m"],
            ai_metrics["MAE_m"],
        )
    )

    outage_rmse_improvement = (
        calculate_improvement(
            outage_dr_metrics["RMSE_m"],
            outage_ai_metrics["RMSE_m"],
        )
    )

    outage_mae_improvement = (
        calculate_improvement(
            outage_dr_metrics["MAE_m"],
            outage_ai_metrics["MAE_m"],
        )
    )

    outage_mask = (
        df["gnss_available"] == False
    )

    results = {
        "data_source": "Stage 8 real pipeline output",

        "total_rows": int(len(df)),

        "outage_samples": int(
            outage_mask.sum()
        ),

        "overall": {
            "dead_reckoning": dr_metrics,
            "ai_corrected": ai_metrics,
            "RMSE_improvement_pct":
                float(overall_rmse_improvement),
            "MAE_improvement_pct":
                float(overall_mae_improvement),
        },

        "gnss_outage": {
            "dead_reckoning":
                outage_dr_metrics,

            "ai_corrected":
                outage_ai_metrics,

            "RMSE_improvement_pct":
                float(outage_rmse_improvement),

            "MAE_improvement_pct":
                float(outage_mae_improvement),
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    return results


# ============================================================
# VALIDATION
# ============================================================

def validate(df, metrics):

    print()
    print("=" * 60)
    print("STAGE 9 VALIDATION")
    print("=" * 60)

    # 1. Row count
    if len(df) > 0:
        print("Input rows: PASS")
    else:
        print("Input rows: FAIL")
        return False

    # 2. Required calculated columns
    required_outputs = [
        "dr_error_m",
        "ai_error_m",
        "dr_error_x",
        "dr_error_y",
        "ai_error_x",
        "ai_error_y",
    ]

    missing = [
        c
        for c in required_outputs
        if c not in df.columns
    ]

    if not missing:
        print("Error columns: PASS")
    else:
        print("Error columns: FAIL")
        print("Missing:", missing)
        return False

    # 3. Error values
    if (
        np.isfinite(df["dr_error_m"]).all()
        and
        np.isfinite(df["ai_error_m"]).all()
    ):
        print("Error values: PASS")
    else:
        print("Error values: FAIL")
        return False

    # 4. AI coordinates
    if (
        np.isfinite(df["x_ai"]).all()
        and
        np.isfinite(df["y_ai"]).all()
    ):
        print("AI corrected coordinates: PASS")
    else:
        print("AI corrected coordinates: FAIL")
        return False

    # 5. Metrics
    if metrics["overall"]["RMSE_improvement_pct"] > 0:
        print("Overall RMSE improvement: PASS")
    else:
        print("Overall RMSE improvement: WARNING")

    outage_improvement = (
        metrics["gnss_outage"]
        ["RMSE_improvement_pct"]
    )

    if outage_improvement > 0:
        print("GNSS outage RMSE improvement: PASS")
    else:
        print(
            "GNSS outage RMSE improvement: WARNING"
        )

    print()
    print("Stage 9 validation: PASS")

    return True


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    df,
    metrics,
):

    overall = metrics["overall"]
    outage = metrics["gnss_outage"]

    dr = overall["dead_reckoning"]
    ai = overall["ai_corrected"]

    outage_dr = outage["dead_reckoning"]
    outage_ai = outage["ai_corrected"]

    print()
    print("=" * 60)
    print("STAGE 9 — REAL DATA EVALUATION")
    print("=" * 60)

    print()
    print("Dataset:")
    print(
        f"Rows: {len(df)}"
    )

    print(
        f"GNSS outage samples: "
        f"{metrics['outage_samples']}"
    )

    print()
    print("OVERALL PERFORMANCE")
    print("-" * 60)

    print(
        f"DR MAE:             "
        f"{dr['MAE_m']:.4f} m"
    )

    print(
        f"AI MAE:             "
        f"{ai['MAE_m']:.4f} m"
    )

    print(
        f"DR RMSE:            "
        f"{dr['RMSE_m']:.4f} m"
    )

    print(
        f"AI RMSE:            "
        f"{ai['RMSE_m']:.4f} m"
    )

    print(
        f"RMSE Improvement:   "
        f"{overall['RMSE_improvement_pct']:.2f}%"
    )

    print(
        f"MAE Improvement:    "
        f"{overall['MAE_improvement_pct']:.2f}%"
    )

    print()
    print("GNSS OUTAGE PERFORMANCE")
    print("-" * 60)

    print(
        f"DR MAE:             "
        f"{outage_dr['MAE_m']:.4f} m"
    )

    print(
        f"AI MAE:             "
        f"{outage_ai['MAE_m']:.4f} m"
    )

    print(
        f"DR RMSE:            "
        f"{outage_dr['RMSE_m']:.4f} m"
    )

    print(
        f"AI RMSE:            "
        f"{outage_ai['RMSE_m']:.4f} m"
    )

    print(
        f"RMSE Improvement:   "
        f"{outage['RMSE_improvement_pct']:.2f}%"
    )

    print(
        f"MAE Improvement:    "
        f"{outage['MAE_improvement_pct']:.2f}%"
    )

    print()
    print("Maximum Errors:")
    print(
        f"DR:                 "
        f"{dr['max_error_m']:.4f} m"
    )

    print(
        f"AI:                 "
        f"{ai['max_error_m']:.4f} m"
    )

    print()
    print("Stage 9 Output:")
    print(
        f"  {ERROR_TIMESERIES_PATH}"
    )

    print(
        f"  {METRICS_PATH}"
    )

    print()
    print("Plots:")
    print(
        f"  {TRAJECTORY_PLOT_PATH}"
    )

    print(
        f"  {ERROR_COMPARISON_PLOT_PATH}"
    )

    print(
        f"  {OUTAGE_ERROR_PLOT_PATH}"
    )

    print()
    print("STAGE 9 RESULT: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 60)
    print("STAGE 9 — DR vs AI-CORRECTED EVALUATION")
    print("=" * 60)

    print()
    print("Loading Stage 8 dataset:")
    print(f"  {INPUT_PATH}")

    df = load_input()

    print(
        f"  Rows: {len(df)}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Calculate errors
    # --------------------------------------------------------

    df = calculate_errors(df)

    print()
    print("Error calculation: PASS")

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    dr_metrics = calculate_metrics(
        df["dr_error_m"]
    )

    ai_metrics = calculate_metrics(
        df["ai_error_m"]
    )

    # --------------------------------------------------------
    # Outage metrics
    # --------------------------------------------------------

    outage_mask = (
        df["gnss_available"] == False
    )

    if outage_mask.any():

        outage_df = df.loc[
            outage_mask
        ]

        outage_dr_metrics = calculate_metrics(
            outage_df["dr_error_m"]
        )

        outage_ai_metrics = calculate_metrics(
            outage_df["ai_error_m"]
        )

    else:

        print()
        print(
            "WARNING: No GNSS outage samples found."
        )

        outage_dr_metrics = calculate_metrics(
            np.array([])
        )

        outage_ai_metrics = calculate_metrics(
            np.array([])
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_error_timeseries(df)

    metrics = save_metrics(
        df,
        dr_metrics,
        ai_metrics,
        outage_dr_metrics,
        outage_ai_metrics,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_trajectory_plot(df)

    make_error_comparison_plot(df)

    make_outage_error_plot(df)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate(
        df,
        metrics,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        df,
        metrics,
    )


if __name__ == "__main__":
    main()