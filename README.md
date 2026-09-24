# Intelligent Dead Reckoning (DR) System — SIH Prototype

AI/ML Based Intelligent Dead Reckoning System for Seamless Navigation.

This repo is being built as a 10-stage hackathon prototype:

1. **Define scenario** ✅ (this stage)
2. Create/obtain trajectory data
3. Convert GPS to X/Y
4. Implement dead reckoning
5. Remove GPS during simulated outage
6. Show DR drift
7. Train ML to predict DR error
8. Correct DR using ML
9. Calculate DR vs AI error
10. Display both trajectories in a dashboard

## Project structure

```
intelligent-dead-reckoning/
├── data/
│   ├── raw/          # raw generated trajectory/sensor data (Stage 2+)
│   ├── processed/     # cleaned / feature-engineered data (later stages)
│   └── models/        # trained ML models (Stage 7+)
├── src/
│   ├── config.py      # centralized scenario configuration (Stage 1)
│   └── scenario.py     # validation, timestamps, GPS-availability helpers (Stage 1)
├── outputs/            # plots, metrics, exported results (later stages)
├── dashboard/          # dashboard app (Stage 10)
├── requirements.txt
└── README.md
```

## Stage 1 — Define scenario

`src/config.py` defines a single `SCENARIO` object (a frozen dataclass)
holding every parameter the rest of the pipeline needs:

- Timing: `TOTAL_DURATION`, `SAMPLING_RATE`
- GPS outage window: `GPS_OUTAGE_START`, `GPS_OUTAGE_END`
- Initial state: latitude, longitude, speed, heading
- Curved road-path shape parameters
- Noise/bias models: GPS noise, accelerometer noise+bias, gyroscope noise+bias
- `RANDOM_SEED` for reproducibility

`src/scenario.py` imports that configuration and:

- validates it (`validate_scenario`)
- computes simulation timestamps (`get_timestamps`)
- provides GPS-availability helpers (`is_gps_available`, `gps_availability_mask`)
- prints a human-readable scenario description (`describe_scenario`)
- runs a smoke test when executed directly

### Run Stage 1

```bash
python src/scenario.py
```

Expected output (abridged):

```
Scenario: Vehicle navigation with simulated GNSS outage
Duration: 120 seconds
Sampling rate: 10 Hz
GPS outage: 60–90 seconds
Total samples: 1200
```

## Setup

```bash
pip install -r requirements.txt
```

## Notes

- Stage 1 does **not** generate any trajectory, run dead reckoning, train
  any ML model, or build the dashboard — those are handled in later stages,
  reusing the same `SCENARIO` config defined here.
- No TensorFlow/PyTorch is used anywhere in this prototype; only lightweight
  dependencies (numpy, pandas, scikit-learn, matplotlib, streamlit).
