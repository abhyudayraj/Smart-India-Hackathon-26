"""
scenario.py
===========
Stage 1 of the Intelligent Dead Reckoning pipeline.

Responsibilities (Stage 1 ONLY):
    1. Import the centralized configuration (config.py).
    2. Validate the scenario parameters.
    3. Calculate the simulation timestamps.
    4. Provide helper functions to check GPS availability at a given time.
    5. Provide a human-readable description of the scenario.
    6. Be directly executable for a quick smoke test.

NOTE: This file does NOT generate trajectory data, does NOT implement dead
reckoning, and does NOT touch ML or dashboarding. Those are later stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow `python src/scenario.py` to work regardless of CWD.
sys.path.append(str(Path(__file__).resolve().parent))

from config import SCENARIO, ScenarioConfig, ensure_directories  # noqa: E402


# --------------------------------------------------------------------------
# 2. Validation
# --------------------------------------------------------------------------
def validate_scenario(cfg: ScenarioConfig = SCENARIO) -> None:
    """
    Validate that the scenario configuration is internally consistent.
    Raises ValueError with a clear message if something is wrong.
    """
    errors = []

    if cfg.TOTAL_DURATION <= 0:
        errors.append("TOTAL_DURATION must be > 0.")

    if cfg.SAMPLING_RATE <= 0:
        errors.append("SAMPLING_RATE must be > 0.")

    if cfg.GPS_OUTAGE_START < 0:
        errors.append("GPS_OUTAGE_START must be >= 0.")

    if cfg.GPS_OUTAGE_END <= cfg.GPS_OUTAGE_START:
        errors.append("GPS_OUTAGE_END must be greater than GPS_OUTAGE_START.")

    if cfg.GPS_OUTAGE_END > cfg.TOTAL_DURATION:
        errors.append("GPS_OUTAGE_END must not exceed TOTAL_DURATION.")

    if not (-90.0 <= cfg.INITIAL_LATITUDE <= 90.0):
        errors.append("INITIAL_LATITUDE must be within [-90, 90] degrees.")

    if not (-180.0 <= cfg.INITIAL_LONGITUDE <= 180.0):
        errors.append("INITIAL_LONGITUDE must be within [-180, 180] degrees.")

    if cfg.INITIAL_SPEED < 0:
        errors.append("INITIAL_SPEED must be >= 0.")

    for name in ("GPS_NOISE_STD_M", "ACCEL_NOISE_STD", "GYRO_NOISE_STD"):
        if getattr(cfg, name) < 0:
            errors.append(f"{name} must be >= 0.")

    if cfg.total_samples <= 0:
        errors.append("Computed total_samples must be > 0.")

    if errors:
        raise ValueError(
            "Invalid scenario configuration:\n  - " + "\n  - ".join(errors)
        )


# --------------------------------------------------------------------------
# 3. Timestamps
# --------------------------------------------------------------------------
def get_timestamps(cfg: ScenarioConfig = SCENARIO) -> np.ndarray:
    """
    Return the array of simulation timestamps (seconds), evenly spaced at
    1/SAMPLING_RATE, covering [0, TOTAL_DURATION).

    Shape: (total_samples,)
    """
    return np.arange(cfg.total_samples) * cfg.dt


# --------------------------------------------------------------------------
# 4. GPS availability helpers
# --------------------------------------------------------------------------
def is_gps_available(t: float, cfg: ScenarioConfig = SCENARIO) -> bool:
    """
    Return True if GPS is available at time `t` (seconds), False during the
    simulated outage window [GPS_OUTAGE_START, GPS_OUTAGE_END).
    """
    return not (cfg.GPS_OUTAGE_START <= t < cfg.GPS_OUTAGE_END)


def gps_availability_mask(timestamps: np.ndarray, cfg: ScenarioConfig = SCENARIO) -> np.ndarray:
    """
    Vectorized version of `is_gps_available` for an array of timestamps.
    Returns a boolean numpy array of the same shape as `timestamps`,
    True where GPS is available.
    """
    outage = (timestamps >= cfg.GPS_OUTAGE_START) & (timestamps < cfg.GPS_OUTAGE_END)
    return ~outage


# --------------------------------------------------------------------------
# 5. Human-readable description
# --------------------------------------------------------------------------
def describe_scenario(cfg: ScenarioConfig = SCENARIO) -> str:
    """Return a short, human-readable summary of the scenario."""
    lines = [
        f"Scenario: {cfg.name}",
        f"Duration: {int(cfg.TOTAL_DURATION)} seconds",
        f"Sampling rate: {int(cfg.SAMPLING_RATE)} Hz",
        f"GPS outage: {int(cfg.GPS_OUTAGE_START)}\u2013{int(cfg.GPS_OUTAGE_END)} seconds",
        f"Total samples: {cfg.total_samples}",
    ]
    return "\n".join(lines)


def describe_scenario_verbose(cfg: ScenarioConfig = SCENARIO) -> str:
    """Return a longer description including noise/bias/init parameters."""
    basic = describe_scenario(cfg)
    extra = [
        "",
        cfg.description,
        "",
        f"Initial position: ({cfg.INITIAL_LATITUDE:.4f}, {cfg.INITIAL_LONGITUDE:.4f})",
        f"Initial speed: {cfg.INITIAL_SPEED} m/s, heading {cfg.INITIAL_HEADING_DEG} deg",
        f"GPS noise std: {cfg.GPS_NOISE_STD_M} m",
        f"Accel noise std: {cfg.ACCEL_NOISE_STD} m/s^2, bias: {cfg.ACCEL_BIAS} m/s^2",
        f"Gyro noise std: {cfg.GYRO_NOISE_STD} rad/s, bias: {cfg.GYRO_BIAS} rad/s",
        f"Random seed: {cfg.RANDOM_SEED}",
        f"Outage duration: {cfg.outage_duration} seconds "
        f"({int(cfg.outage_duration * cfg.SAMPLING_RATE)} samples)",
    ]
    return basic + "\n".join(extra)


# --------------------------------------------------------------------------
# 6. Direct execution / smoke test
# --------------------------------------------------------------------------
def main() -> None:
    ensure_directories()
    validate_scenario(SCENARIO)

    timestamps = get_timestamps(SCENARIO)
    mask = gps_availability_mask(timestamps, SCENARIO)
    n_outage_samples = int((~mask).sum())

    print(describe_scenario(SCENARIO))
    print()
    print(f"Validation: OK (all scenario parameters are consistent)")
    print(f"Timestamps: first={timestamps[0]:.2f}s, last={timestamps[-1]:.2f}s, "
          f"count={len(timestamps)}")
    print(f"GPS outage samples: {n_outage_samples} "
          f"({n_outage_samples / len(timestamps) * 100:.1f}% of trip)")
    print(f"GPS available at t=0s?  {is_gps_available(0.0, SCENARIO)}")
    print(f"GPS available at t=75s? {is_gps_available(75.0, SCENARIO)}")
    print(f"GPS available at t=100s? {is_gps_available(100.0, SCENARIO)}")


if __name__ == "__main__":
    main()
