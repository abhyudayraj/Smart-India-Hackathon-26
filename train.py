"""
STAGE 7 — ML Dead Reckoning Error Prediction

Trains Random Forest models using the real Stage 6 dataset.

Input:
    stage6/data/drift_analysis.csv

Outputs:
    models/error_x_model.pkl
    models/error_y_model.pkl
    models/feature_config.json

The models predict:
    error_x
    error_y

These predictions will be consumed by Stage 8.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.features import (
    FEATURE_COLUMNS,
    make_targets,
    prepare_features,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

STAGE6_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "stage6",
    "data",
    "drift_analysis.csv",
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

ERROR_X_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "error_x_model.pkl",
)

ERROR_Y_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "error_y_model.pkl",
)

FEATURE_CONFIG_PATH = os.path.join(
    MODELS_DIR,
    "feature_config.json",
)


# ============================================================
# LOAD REAL STAGE 6 DATA
# ============================================================

def load_stage6_data():

    if not os.path.exists(
        STAGE6_DATA_PATH
    ):

        raise FileNotFoundError(
            "Stage 6 dataset not found:\n"
            f"{STAGE6_DATA_PATH}\n\n"
            "Run Stage 6 before Stage 7."
        )

    df = pd.read_csv(
        STAGE6_DATA_PATH
    )

    print(
        "Loaded Stage 6 dataset:"
    )

    print(
        f"  {STAGE6_DATA_PATH}"
    )

    print(
        f"  Rows: {len(df)}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_split(
    df,
    train_fraction=0.8,
):

    n = len(df)

    split_index = int(
        n * train_fraction
    )

    train_df = (
        df.iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_df,
        test_df,
    )


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    df,
):

    train_df, test_df = chronological_split(
        df,
        train_fraction=0.8,
    )

    print()
    print(
        "Chronological train/test split:"
    )

    print(
        f"  Training rows: {len(train_df)}"
    )

    print(
        f"  Testing rows:  {len(test_df)}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = prepare_features(
        train_df
    )

    X_test = prepare_features(
        test_df
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    y_train = make_targets(
        train_df
    )

    y_test = make_targets(
        test_df
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    forbidden = [
        "ground_truth_x_m",
        "ground_truth_y_m",
        "dr_x_m",
        "dr_y_m",
        "position_error_m",
        "dr_error_m",
        "error_x",
        "error_y",
    ]

    leaked = [
        c
        for c in forbidden
        if c in X_train.columns
    ]

    if leaked:

        raise AssertionError(
            "DATA LEAKAGE DETECTED: "
            f"{leaked}"
        )

    print()
    print(
        "Feature leakage check: PASS"
    )

    print(
        "Features used:"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    model_x = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model_y = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print()
    print(
        "Training error_x model..."
    )

    model_x.fit(
        X_train,
        y_train["error_x"],
    )

    print(
        "Training error_y model..."
    )

    model_y.fit(
        X_train,
        y_train["error_y"],
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred_x = model_x.predict(
        X_test
    )

    pred_y = model_y.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse_x = float(
        np.sqrt(
            mean_squared_error(
                y_test["error_x"],
                pred_x,
            )
        )
    )

    rmse_y = float(
        np.sqrt(
            mean_squared_error(
                y_test["error_y"],
                pred_y,
            )
        )
    )

    mae_x = float(
        np.mean(
            np.abs(
                y_test["error_x"].to_numpy()
                - pred_x
            )
        )
    )

    mae_y = float(
        np.mean(
            np.abs(
                y_test["error_y"].to_numpy()
                - pred_y
            )
        )
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model_x,
        ERROR_X_MODEL_PATH,
    )

    joblib.dump(
        model_y,
        ERROR_Y_MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save feature configuration
    # --------------------------------------------------------

    feature_config = {

        "feature_columns":
            FEATURE_COLUMNS,

        "model_type":
            "RandomForestRegressor",

        "training_source":
            "stage6/data/drift_analysis.csv",

        "split_method":
            "chronological_80_20",

        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        "rmse_x":
            rmse_x,

        "rmse_y":
            rmse_y,

        "mae_x":
            mae_x,

        "mae_y":
            mae_y,
    }

    with open(
        FEATURE_CONFIG_PATH,
        "w",
    ) as f:

        json.dump(
            feature_config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        "rmse_x":
            rmse_x,

        "rmse_y":
            rmse_y,

        "mae_x":
            mae_x,

        "mae_y":
            mae_y,

        "pred_x":
            pred_x,

        "pred_y":
            pred_y,

        "y_test":
            y_test,
    }


# ============================================================
# PREDICTION FUNCTION FOR STAGE 8
# ============================================================

def predict_error(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if not (
        os.path.exists(
            ERROR_X_MODEL_PATH
        )
        and
        os.path.exists(
            ERROR_Y_MODEL_PATH
        )
    ):

        raise FileNotFoundError(
            "Stage 7 models not found. "
            "Run python -m src.train first."
        )

    model_x = joblib.load(
        ERROR_X_MODEL_PATH
    )

    model_y = joblib.load(
        ERROR_Y_MODEL_PATH
    )

    X = prepare_features(
        df
    )

    result = pd.DataFrame(
        index=df.index
    )

    result[
        "predicted_error_x"
    ] = model_x.predict(X)

    result[
        "predicted_error_y"
    ] = model_y.predict(X)

    return result


# ============================================================
# SELF VALIDATION
# ============================================================

def validate_models(
    df,
    results,
):

    print()
    print(
        "===================================================="
    )
    print(
        "STAGE 7 VALIDATION"
    )
    print(
        "===================================================="
    )

    # --------------------------------------------------------
    # Model files
    # --------------------------------------------------------

    models_exist = (
        os.path.exists(
            ERROR_X_MODEL_PATH
        )
        and
        os.path.exists(
            ERROR_Y_MODEL_PATH
        )
    )

    print(
        "Model files saved:",
        "PASS" if models_exist else "FAIL",
    )

    if not models_exist:
        return False

    # --------------------------------------------------------
    # Feature configuration
    # --------------------------------------------------------

    config_exists = os.path.exists(
        FEATURE_CONFIG_PATH
    )

    print(
        "Feature configuration saved:",
        "PASS" if config_exists else "FAIL",
    )

    if not config_exists:
        return False

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    features = prepare_features(
        df
    )

    correct_columns = (
        list(features.columns)
        == FEATURE_COLUMNS
    )

    print(
        "Feature columns:",
        "PASS" if correct_columns else "FAIL",
    )

    if not correct_columns:
        return False

    # --------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------

    predictions = predict_error(
        df.head(20)
    )

    expected_prediction_columns = [
        "predicted_error_x",
        "predicted_error_y",
    ]

    prediction_columns_ok = (
        list(predictions.columns)
        == expected_prediction_columns
    )

    print(
        "Prediction interface:",
        "PASS"
        if prediction_columns_ok
        else "FAIL",
    )

    if not prediction_columns_ok:
        return False

    if predictions.isna().any().any():

        print(
            "Prediction values: FAIL"
        )

        return False

    print(
        "Prediction values: PASS"
    )

    print()
    print(
        "Stage 7 validation: PASS"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "===================================================="
    )
    print(
        "STAGE 7 — ML ERROR PREDICTION"
    )
    print(
        "===================================================="
    )

    df = load_stage6_data()

    results = train_models(
        df
    )

    print()
    print(
        "===================================================="
    )
    print(
        "MODEL PERFORMANCE"
    )
    print(
        "===================================================="
    )

    print(
        f"Test RMSE error_x: "
        f"{results['rmse_x']:.4f} m"
    )

    print(
        f"Test RMSE error_y: "
        f"{results['rmse_y']:.4f} m"
    )

    print(
        f"Test MAE error_x:  "
        f"{results['mae_x']:.4f} m"
    )

    print(
        f"Test MAE error_y:  "
        f"{results['mae_y']:.4f} m"
    )

    print()
    print(
        "Saved models:"
    )

    print(
        f"  {ERROR_X_MODEL_PATH}"
    )

    print(
        f"  {ERROR_Y_MODEL_PATH}"
    )

    print(
        f"  {FEATURE_CONFIG_PATH}"
    )

    validation_passed = validate_models(
        df,
        results,
    )

    print()

    if not validation_passed:

        print(
            "STAGE 7 RESULT: FAIL"
        )

        sys.exit(1)

    print(
        "STAGE 7 RESULT: PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()