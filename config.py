"""
config.py
=========
Centralized configuration for the Intelligent Dead Reckoning (DR) prototype.

Every later stage of the pipeline (trajectory generation, GPS/IMU simulation,
dead reckoning, ML error correction, dashboarding, ...) imports the SAME
`SCENARIO` object from this file so that all stages stay perfectly in sync.

Stage 1 responsibility: define parameters ONLY. No trajectory / simulation
logic lives here.
"""

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Project paths (shared by all stages)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Single source of truth for all scenario / simulation parameters.

    Frozen (immutable) so that no later stage can accidentally mutate
    shared configuration at runtime.
    """

    # ---------------- Scenario description ----------------
    name: str = "Vehicle navigation with simulated GNSS outage"
    description: str = (
        "A vehicle travels along a curved road-like path while its position "
        "is tracked using GPS and an onboard IMU (accelerometer + gyroscope). "
        "GPS signal is lost for a fixed window in the middle of the trip, "
        "during which the system must rely on dead reckoning (and later, "
        "ML-based correction) to keep estimating position."
    )

    # ---------------- Timing ----------------
    TOTAL_DURATION: float = 120.0      # seconds
    SAMPLING_RATE: float = 10.0        # Hz (IMU/GPS sample rate)

    # ---------------- GPS outage window ----------------
    GPS_OUTAGE_START: float = 60.0     # seconds
    GPS_OUTAGE_END: float = 90.0       # seconds

    # ---------------- Initial vehicle state ----------------
    INITIAL_LATITUDE: float = 28.6139      # degrees (New Delhi, arbitrary demo origin)
    INITIAL_LONGITUDE: float = 77.2090     # degrees
    INITIAL_SPEED: float = 10.0            # m/s (~36 km/h)
    INITIAL_HEADING_DEG: float = 0.0       # degrees, 0 = North, clockwise positive

    # ---------------- Road / path shape (curved road-like path) ----------------
    # Used by Stage 2 to generate a smooth curved trajectory (e.g. sine-perturbed
    # heading or a series of arcs). Kept here so the shape is configurable in
    # one place.
    PATH_CURVE_AMPLITUDE_DEG: float = 25.0     # max heading swing (degrees)
    PATH_CURVE_FREQUENCY_HZ: float = 0.02      # how fast the curve oscillates

    # ---------------- GPS noise model ----------------
    GPS_NOISE_STD_M: float = 3.0           # 1-sigma GPS position noise (meters)

    # ---------------- IMU noise & bias model ----------------
    ACCEL_NOISE_STD: float = 0.15          # m/s^2, 1-sigma accelerometer noise
    GYRO_NOISE_STD: float = 0.02           # rad/s, 1-sigma gyroscope noise

    ACCEL_BIAS: float = 0.05               # m/s^2, constant accelerometer bias
    GYRO_BIAS: float = 0.01                # rad/s, constant gyroscope bias

    # ---------------- Reproducibility ----------------
    RANDOM_SEED: int = 42

    # ---------------- Derived / bookkeeping (not user-editable in practice) ----------------
    units: dict = field(default_factory=lambda: {
        "duration": "seconds",
        "sampling_rate": "Hz",
        "lat_lon": "degrees",
        "speed": "m/s",
        "noise_std_gps": "meters",
        "noise_std_accel": "m/s^2",
        "noise_std_gyro": "rad/s",
    })

    # ---------------- Convenience properties ----------------
    @property
    def dt(self) -> float:
        """Time step between samples, in seconds."""
        return 1.0 / self.SAMPLING_RATE

    @property
    def total_samples(self) -> int:
        """Total number of samples over the full simulation duration."""
        return int(round(self.TOTAL_DURATION * self.SAMPLING_RATE))

    @property
    def outage_duration(self) -> float:
        """Length of the GPS outage window, in seconds."""
        return self.GPS_OUTAGE_END - self.GPS_OUTAGE_START


# A single shared instance every stage should import:
#     from config import SCENARIO
SCENARIO = ScenarioConfig()


def ensure_directories() -> None:
    """Create all project directories if they do not already exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, DASHBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick manual sanity check when running config.py directly.
    ensure_directories()
    print("Project root:", PROJECT_ROOT)
    print(SCENARIO)
