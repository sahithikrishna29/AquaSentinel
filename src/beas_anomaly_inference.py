
"""
Beas River Anomaly Detection - API Inference
Uses the existing trained pH and Dissolved Oxygen Isolation Forest model.
Does not retrain or modify existing model files.
"""

from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "river_beas_v2"

MODEL_PATH = MODEL_DIR / "beas_ph_do_anomaly_model.joblib"
SCALER_PATH = MODEL_DIR / "beas_ph_do_anomaly_scaler.joblib"
METADATA_PATH = MODEL_DIR / "beas_ph_do_anomaly_metadata.joblib"


class BeasAnomalyInference:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.loaded = False

    def load(self):
        """Load the existing trained Beas anomaly model."""

        for path in [MODEL_PATH, SCALER_PATH, METADATA_PATH]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required Beas anomaly file not found: {path}"
                )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.metadata = joblib.load(METADATA_PATH)

        expected_features = ["pH", "Dissolved Oxygen"]

        if self.metadata.get("features") != expected_features:
            raise ValueError(
                "Unexpected anomaly model features: "
                f"{self.metadata.get('features')}"
            )

        if getattr(self.scaler, "n_features_in_", None) != 2:
            raise ValueError("Anomaly scaler must accept two features.")

        self.loaded = True
        print("Beas anomaly inference model loaded successfully.")

    def detect(self, readings):
        """Detect anomalies using pH and Dissolved Oxygen readings."""

        if not self.loaded:
            self.load()

        ph = readings.get("ph", readings.get("pH"))
        do = readings.get(
            "do",
            readings.get("Dissolved Oxygen")
        )

        if ph is None or do is None:
            raise ValueError(
                "Both pH and Dissolved Oxygen readings are required."
            )

        try:
            values = np.array([[float(ph), float(do)]], dtype=float)
        except (TypeError, ValueError):
            raise ValueError("pH and Dissolved Oxygen must be numeric.")

        if not np.isfinite(values).all():
            raise ValueError("Sensor readings must be finite numbers.")

        scaled_values = self.scaler.transform(values)

        prediction = int(self.model.predict(scaled_values)[0])
        score = float(
            self.model.decision_function(scaled_values)[0]
        )

        is_anomaly = prediction == -1

        return {
            "model": "Beas River Isolation Forest",
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "anomaly_score": round(score, 4),
            "input_features": {
                "pH": float(ph),
                "Dissolved Oxygen": float(do)
            }
        }


beas_anomaly_detector = BeasAnomalyInference()
