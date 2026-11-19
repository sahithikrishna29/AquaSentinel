
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "river_beas_v2"

MODEL_PATH = MODEL_DIR / "xgboost_beas_model.joblib"
SCALER_PATH = MODEL_DIR / "beas_scaler.joblib"
META_PATH = MODEL_DIR / "beas_feature_meta.joblib"


# Load existing saved files
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
metadata = joblib.load(META_PATH)

FEATURE_COLUMNS = metadata["feature_columns"]
CLASS_MAPPING = metadata["class_mapping"]


def predict_water_quality(readings):
    """
    Predict water quality using five required water parameters.

    readings must be a dictionary containing all five features.
    """

    # Check missing features
    missing = [
        feature for feature in FEATURE_COLUMNS
        if feature not in readings
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    # Prepare input in the exact training feature order
    values = {}

    for feature in FEATURE_COLUMNS:
        try:
            value = float(readings[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid number"
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be finite"
            )

        values[feature] = value

    input_data = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    # Scale the input
    scaled_data = scaler.transform(input_data)

    # Predict class
    prediction = int(model.predict(scaled_data)[0])

    # Calculate class probabilities
    probabilities = model.predict_proba(scaled_data)[0]

    probability_results = {
        CLASS_MAPPING[int(class_id)]: round(float(prob), 4)
        for class_id, prob in zip(model.classes_, probabilities)
    }

    return {
        "prediction": CLASS_MAPPING[prediction],
        "class_id": prediction,
        "probabilities": probability_results
    }
    
