# AI/ML Based Intelligent Dead Reckoning System for Seamless Navigation

## Stage 2 â€” Synthetic Trajectory Generation

**Why synthetic data?** Real GNSS/IMU logs with known ground truth and a
labeled GPS outage are hard to get on short notice. Generating synthetic
data lets us control exactly when GPS is available, guarantees a known
ground truth for every later error calculation, and keeps the dataset
reproducible (fixed random seed = 42).

- **Ground truth**: the true, noise-free position/speed/heading/
  acceleration/angular velocity of the simulated vehicle. This is the
  reference every later stage compares against.
- **GPS measurements**: a noisy version of the ground-truth position
  (`gps_x`, `gps_y`, `gps_latitude`, `gps_longitude`) plus a
  `gps_available` flag. During the simulated outage (60sâ€“90s) these
  columns are `NaN` and `gps_available = 0`, while timestamps stay
  continuous.
- **IMU measurements**: simulated accelerometer (`accel_x`, `accel_y`)
  and gyroscope (`gyro_z`) readings with small noise **and** small
  constant biases. The biases are what make a future dead-reckoning
  integrator drift over time, especially while GPS is unavailable.
- **Why simulate a GPS outage?** The whole point of the project is to
  show that dead reckoning drifts during an outage and that an ML model
  can learn to correct that drift. Stage 2 is what makes that outage
  possible to study, because we know the true path even when GPS doesn't.

**How later stages use this dataset:**
- Stage 3 converts `gps_latitude/longitude` to local X/Y.
- Stage 4 integrates `accel_x/accel_y/gyro_z` to perform dead reckoning.
- Stage 5 handles the logic around the GPS outage window.
- Stage 6 compares the DR path against `ground_x/ground_y` to show drift.
- Stage 7 trains an ML model to predict DR error using this data.
- Stage 8 uses that model to correct the DR trajectory.
- Stage 9 compares DR error vs AI-corrected error.
- Stage 10 displays everything on a dashboard.

Regenerate the dataset with:
```
python src/trajectory.py
```

## Stage 2 — Synthetic Trajectory Generation

**Why synthetic data?** Real GNSS/IMU logs with known ground truth and a
labeled GPS outage are hard to get on short notice. Generating synthetic
data lets us control exactly when GPS is available, guarantees a known
ground truth for every later error calculation, and keeps the dataset
reproducible (fixed random seed = 42).

- **Ground truth**: the true, noise-free position/speed/heading/
  acceleration/angular velocity of the simulated vehicle. This is the
  reference every later stage compares against.
- **GPS measurements**: a noisy version of the ground-truth position
  (`gps_x`, `gps_y`, `gps_latitude`, `gps_longitude`) plus a
  `gps_available` flag. During the simulated outage (60s–90s) these
  columns are `NaN` and `gps_available = 0`, while timestamps stay
  continuous.
- **IMU measurements**: simulated accelerometer (`accel_x`, `accel_y`)
  and gyroscope (`gyro_z`) readings with small noise **and** small
  constant biases. The biases are what make a future dead-reckoning
  integrator drift over time, especially while GPS is unavailable.
- **Why simulate a GPS outage?** The whole point of the project is to
  show that dead reckoning drifts during an outage and that an ML model
  can learn to correct that drift. Stage 2 is what makes that outage
  possible to study, because we know the true path even when GPS doesn't.

**How later stages use this dataset:**
- Stage 3 converts `gps_latitude/longitude` to local X/Y.
- Stage 4 integrates `accel_x/accel_y/gyro_z` to perform dead reckoning.
- Stage 5 handles the logic around the GPS outage window.
- Stage 6 compares the DR path against `ground_x/ground_y` to show drift.
- Stage 7 trains an ML model to predict DR error using this data.
- Stage 8 uses that model to correct the DR trajectory.
- Stage 9 compares DR error vs AI-corrected error.
- Stage 10 displays everything on a dashboard.

Regenerate the dataset with:
```
python src/trajectory.py
```
