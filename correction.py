"""
Shared Stage 8 ML correction module.

This module loads the Stage 7 models and applies their predicted
DR position errors to the Stage 6 dead-reckoning trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

ERROR_X_MODEL_PATH = MODELS_DIR / "error_x_model.pkl"
ERROR_Y_MODEL_PATH = MODELS_DIR / "error_y_model.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"


def _load_models_and_feature_columns():

    required_paths = [
        ERROR_X_MODEL_PATH,
        ERROR_Y_MODEL_PATH,
        FEATURE_CONFIG_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing Stage 7 artifact(s): {missing}"
        )

    with FEATURE_CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    config_features = config.get("feature_columns")

    if config_features != FEATURE_COLUMNS:
        raise ValueError(
            "feature_config.json feature order does not match "
            "src.features.FEATURE_COLUMNS"
        )

    model_x = joblib.load(ERROR_X_MODEL_PATH)
    model_y = joblib.load(ERROR_Y_MODEL_PATH)

    for name, model in [
        ("error_x", model_x),
        ("error_y", model_y),
    ]:

        model_features = list(
            getattr(model, "feature_names_in_", [])
        )

        if model_features and model_features != config_features:

            raise ValueError(
                f"{name} model feature order does not match "
                "feature_config.json"
            )

    return model_x, model_y, config_features


def correct_dead_reckoning(
    df: pd.DataFrame
) -> pd.DataFrame:

    required = [
        "x_dr",
        "y_dr",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "correct_dead_reckoning: missing required columns: "
            f"{missing}"
        )

    model_x, model_y, feature_columns = (
        _load_models_and_feature_columns()
    )

    features = prepare_features(
        df
    ).loc[:, feature_columns]

    result = df.copy()

    result["predicted_error_x"] = (
        model_x.predict(features)
    )

    result["predicted_error_y"] = (
        model_y.predict(features)
    )

    result["x_ai"] = (
        result["x_dr"]
        + result["predicted_error_x"]
    )

    result["y_ai"] = (
        result["y_dr"]
        + result["predicted_error_y"]
    )

    return result