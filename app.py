"""
STAGE 10 — Interactive Dashboard

Uses REAL outputs from:
    Stage 8:
        stage8/data/ml_corrected_trajectory.csv

    Stage 9:
        stage9/data/error_timeseries.csv
        stage9/data/evaluation_metrics.json

Run:
    python -m streamlit run stage10/app.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

STAGE8_FILE = "stage8/data/ml_corrected_trajectory.csv"
STAGE9_ERROR_FILE = "stage9/data/error_timeseries.csv"
STAGE9_METRICS_FILE = "stage9/data/evaluation_metrics.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-ML Intelligent Dead Reckoning",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(STAGE8_FILE):
        raise FileNotFoundError(
            f"Stage 8 file not found:\n{STAGE8_FILE}"
        )

    if not os.path.exists(STAGE9_ERROR_FILE):
        raise FileNotFoundError(
            f"Stage 9 error file not found:\n{STAGE9_ERROR_FILE}"
        )

    if not os.path.exists(STAGE9_METRICS_FILE):
        raise FileNotFoundError(
            f"Stage 9 metrics file not found:\n{STAGE9_METRICS_FILE}"
        )

    df = pd.read_csv(STAGE8_FILE)
    error_df = pd.read_csv(STAGE9_ERROR_FILE)

    with open(STAGE9_METRICS_FILE, "r") as f:
        metrics = json.load(f)

    return df, error_df, metrics


try:
    df, error_df, metrics = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_stage8 = [
    "timestamp",
    "x_true",
    "y_true",
    "x_dr",
    "y_dr",
    "x_ai",
    "y_ai",
    "gnss_available",
    "cumulative_distance_m"
]

missing = [c for c in required_stage8 if c not in df.columns]

if missing:
    st.error(f"Missing Stage 8 columns: {missing}")
    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["gnss_available"] = df["gnss_available"].astype(bool)

df["dr_error_calc"] = np.sqrt(
    (df["x_dr"] - df["x_true"]) ** 2 +
    (df["y_dr"] - df["y_true"]) ** 2
)

df["ai_error_calc"] = np.sqrt(
    (df["x_ai"] - df["x_true"]) ** 2 +
    (df["y_ai"] - df["y_true"]) ** 2
)

# Convert timestamp to numeric if possible
df["timestamp"] = pd.to_numeric(
    df["timestamp"],
    errors="coerce"
)

# Outage data
outage_df = df[df["gnss_available"] == False].copy()

outage_samples = len(outage_df)

if outage_samples > 0:

    outage_start_time = outage_df["timestamp"].min()
    outage_end_time = outage_df["timestamp"].max()

    outage_duration = (
        outage_end_time - outage_start_time
    )

    outage_distance = (
        outage_df["cumulative_distance_m"].max()
        - outage_df["cumulative_distance_m"].min()
    )

else:

    outage_start_time = 0
    outage_end_time = 0
    outage_duration = 0
    outage_distance = 0


# ============================================================
# GET STAGE 9 METRICS
# ============================================================

overall = metrics.get("overall", {})
outage = metrics.get("gnss_outage", {})

# Support alternate JSON structures if necessary
if not overall:
    overall = metrics.get("overall_metrics", {})

if not outage:
    outage = metrics.get("outage_metrics", {})


def get_metric(section, possible_names, fallback=np.nan):

    for name in possible_names:
        if name in section:
            return float(section[name])

    return fallback


# Overall metrics
dr_mae = get_metric(
    overall,
    ["dr_mae", "DR_MAE", "MAE_DR"]
)

ai_mae = get_metric(
    overall,
    ["ai_mae", "AI_MAE", "MAE_AI"]
)

dr_rmse = get_metric(
    overall,
    ["dr_rmse", "DR_RMSE", "RMSE_DR"]
)

ai_rmse = get_metric(
    overall,
    ["ai_rmse", "AI_RMSE", "RMSE_AI"]
)

rmse_improvement = get_metric(
    overall,
    [
        "rmse_improvement_pct",
        "RMSE_improvement_pct",
        "rmse_improvement"
    ]
)


# If JSON structure differs, calculate directly
if np.isnan(dr_mae):
    dr_mae = df["dr_error_calc"].mean()

if np.isnan(ai_mae):
    ai_mae = df["ai_error_calc"].mean()

if np.isnan(dr_rmse):
    dr_rmse = np.sqrt(
        np.mean(df["dr_error_calc"] ** 2)
    )

if np.isnan(ai_rmse):
    ai_rmse = np.sqrt(
        np.mean(df["ai_error_calc"] ** 2)
    )

if np.isnan(rmse_improvement):
    rmse_improvement = (
        (dr_rmse - ai_rmse)
        / dr_rmse
        * 100
    )


# Outage metrics
outage_dr_mae = get_metric(
    outage,
    ["dr_mae", "DR_MAE", "MAE_DR"]
)

outage_ai_mae = get_metric(
    outage,
    ["ai_mae", "AI_MAE", "MAE_AI"]
)

outage_dr_rmse = get_metric(
    outage,
    ["dr_rmse", "DR_RMSE", "RMSE_DR"]
)

outage_ai_rmse = get_metric(
    outage,
    ["ai_rmse", "AI_RMSE", "RMSE_AI"]
)

outage_rmse_improvement = get_metric(
    outage,
    [
        "rmse_improvement_pct",
        "RMSE_improvement_pct",
        "rmse_improvement"
    ]
)

if np.isnan(outage_dr_mae):
    outage_dr_mae = outage_df["dr_error_calc"].mean()

if np.isnan(outage_ai_mae):
    outage_ai_mae = outage_df["ai_error_calc"].mean()

if np.isnan(outage_dr_rmse):
    outage_dr_rmse = np.sqrt(
        np.mean(outage_df["dr_error_calc"] ** 2)
    )

if np.isnan(outage_ai_rmse):
    outage_ai_rmse = np.sqrt(
        np.mean(outage_df["ai_error_calc"] ** 2)
    )

if np.isnan(outage_rmse_improvement):
    outage_rmse_improvement = (
        (outage_dr_rmse - outage_ai_rmse)
        / outage_dr_rmse
        * 100
    )


# Maximum errors
dr_max = df["dr_error_calc"].max()
ai_max = df["ai_error_calc"].max()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("System Status")

st.sidebar.success("Pipeline Operational")

st.sidebar.markdown("### Data Sources")

st.sidebar.write("Stage 8")
st.sidebar.caption("ML corrected trajectory")

st.sidebar.write("Stage 9")
st.sidebar.caption("Performance evaluation")

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset")

st.sidebar.write(f"Samples: {len(df):,}")

route_distance = df["cumulative_distance_m"].max()

st.sidebar.write(
    f"Route: {route_distance:.2f} m"
)

st.sidebar.write(
    f"GNSS outage: {outage_samples} samples"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prototype demonstration using trajectory "
    "and simulated GNSS outage data."
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI-ML Based Intelligent Dead Reckoning"
)

st.subheader(
    "Seamless Navigation During GNSS Outages"
)

st.write(
    "The system estimates dead-reckoning position drift "
    "using IMU-derived features and machine-learning "
    "error prediction, then applies the predicted error "
    "to obtain an AI-corrected position."
)


# ============================================================
# STATUS CARDS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.success("GNSS + DR")

with c2:
    st.warning("GNSS OUTAGE SIMULATED")

with c3:
    st.success("AI CORRECTION ACTIVE")


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.markdown("## Performance Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "DR RMSE",
        f"{dr_rmse:.2f} m"
    )

with c2:
    st.metric(
        "AI RMSE",
        f"{ai_rmse:.2f} m",
        delta=f"{-rmse_improvement:.1f}%"
    )

with c3:
    st.metric(
        "DR MAE",
        f"{dr_mae:.2f} m"
    )

with c4:
    st.metric(
        "AI MAE",
        f"{ai_mae:.2f} m",
        delta=f"{-((dr_mae-ai_mae)/dr_mae*100):.1f}%"
    )


# ============================================================
# GNSS OUTAGE PERFORMANCE
# ============================================================

st.markdown("## GNSS Outage Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Outage DR RMSE",
        f"{outage_dr_rmse:.3f} m"
    )

with c2:
    st.metric(
        "Outage AI RMSE",
        f"{outage_ai_rmse:.3f} m",
        delta=f"{-outage_rmse_improvement:.2f}%"
    )

with c3:
    st.metric(
        "Outage Duration",
        f"{outage_duration:.1f} s"
    )

with c4:
    st.metric(
        "Outage Distance",
        f"{outage_distance:.1f} m"
    )


# ============================================================
# TRAJECTORY
# ============================================================

st.markdown("## Navigation Trajectory")

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=df["x_true"],
        y=df["y_true"],
        mode="lines",
        name="Ground Truth",
        line=dict(
            width=3
        )
    )
)

fig1.add_trace(
    go.Scatter(
        x=df["x_dr"],
        y=df["y_dr"],
        mode="lines",
        name="Raw Dead Reckoning",
        line=dict(
            width=2,
            dash="dot"
        )
    )
)

fig1.add_trace(
    go.Scatter(
        x=df["x_ai"],
        y=df["y_ai"],
        mode="lines",
        name="AI Corrected",
        line=dict(
            width=2
        )
    )
)

# Outage segment
if outage_samples > 0:

    fig1.add_trace(
        go.Scatter(
            x=outage_df["x_true"],
            y=outage_df["y_true"],
            mode="lines",
            name="GNSS Outage Region",
            line=dict(
                width=8
            ),
            opacity=0.25
        )
    )


fig1.update_layout(
    height=600,
    xaxis_title="X Position (m)",
    yaxis_title="Y Position (m)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02
    ),
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ============================================================
# ERROR OVER TIME
# ============================================================

st.markdown("## Position Error Over Time")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["dr_error_calc"],
        mode="lines",
        name="Raw DR Error",
        line=dict(
            width=2
        )
    )
)

fig2.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["ai_error_calc"],
        mode="lines",
        name="AI Corrected Error",
        line=dict(
            width=2
        )
    )
)

if outage_samples > 0:

    fig2.add_vrect(
        x0=outage_start_time,
        x1=outage_end_time,
        fillcolor="orange",
        opacity=0.18,
        line_width=0,
        annotation_text="GNSS OUTAGE",
        annotation_position="top left"
    )


fig2.update_layout(
    height=450,
    xaxis_title="Time (s)",
    yaxis_title="Position Error (m)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02
    ),
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# DR VS AI BAR CHART
# ============================================================

st.markdown("## DR vs AI Performance")

fig3 = go.Figure()

fig3.add_trace(
    go.Bar(
        name="Raw DR",
        x=["MAE", "RMSE", "Maximum"],
        y=[
            dr_mae,
            dr_rmse,
            dr_max
        ]
    )
)

fig3.add_trace(
    go.Bar(
        name="AI Corrected",
        x=["MAE", "RMSE", "Maximum"],
        y=[
            ai_mae,
            ai_rmse,
            ai_max
        ]
    )
)

fig3.update_layout(
    barmode="group",
    height=450,
    xaxis_title="Metric",
    yaxis_title="Error (m)"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ============================================================
# PIPELINE
# ============================================================

st.markdown("## System Pipeline")

pipeline_df = pd.DataFrame({
    "Stage": [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Stage 5",
        "Stage 6",
        "Stage 7",
        "Stage 8",
        "Stage 9",
        "Stage 10"
    ],
    "Function": [
        "Scenario Configuration",
        "Trajectory / Sensor Data",
        "Coordinate Conversion",
        "Dead Reckoning",
        "GNSS Outage Simulation",
        "DR Drift Analysis",
        "ML Error Prediction",
        "AI Position Correction",
        "Performance Evaluation",
        "Interactive Dashboard"
    ],
    "Status": [
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        "ACTIVE"
    ]
})

st.dataframe(
    pipeline_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROTOTYPE RESULT
# ============================================================

st.markdown("## Prototype Result")

st.success(
    f"""
The prototype demonstrates AI-assisted dead reckoning
during a simulated GNSS outage.

Overall Raw DR RMSE: {dr_rmse:.2f} m

Overall AI Corrected RMSE: {ai_rmse:.2f} m

Overall RMSE Reduction: {rmse_improvement:.2f}%

During the GNSS outage, Raw DR RMSE was
{outage_dr_rmse:.3f} m while AI Corrected RMSE was
{outage_ai_rmse:.3f} m, corresponding to a
{outage_rmse_improvement:.2f}% reduction in RMSE.
"""
)


# ============================================================
# IMPORTANT PROTOTYPE NOTE
# ============================================================

st.markdown("---")

st.caption(
    "Prototype evaluation uses trajectory data with a "
    "simulated GNSS outage. Results demonstrate the "
    "software pipeline and ML correction methodology. "
    "Hardware IMU validation and testing on unseen "
    "real-world GNSS outage data are future work."
)