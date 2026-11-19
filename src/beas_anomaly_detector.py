
"""
Beas River Anomaly Detection
Member 2 - AquaSentinel

Uses paired historical pH and Dissolved Oxygen observations.
Does not modify the existing classification model or scaler.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "beas_observations_only.csv"

MODEL_DIR = ROOT / "models" / "river_beas_v2"
RESULT_DIR = ROOT / "Results" / "beas_anomaly_detection"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["pH", "Dissolved Oxygen"]


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    df = df[
        df["parameter"].isin(FEATURES)
        & df["record_type"].isin([
            "sample_observation",
            "monthly_observation"
        ])
    ].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    pivot = df.pivot_table(
        index=[
            "station_name",
            "date",
            "date_precision",
            "record_type"
        ],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()

    pivot = pivot.dropna(subset=FEATURES).copy()

    if len(pivot) < 20:
        raise ValueError("Not enough paired observations to train.")

    return pivot


def train_detector(data):
    X = data[FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    results = data.copy()
    results["anomaly_prediction"] = predictions
    results["anomaly_score"] = scores
    results["anomaly_status"] = np.where(
        predictions == -1,
        "ANOMALY",
        "NORMAL"
    )

    joblib.dump(
        model,
        MODEL_DIR / "beas_ph_do_anomaly_model.joblib"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "beas_ph_do_anomaly_scaler.joblib"
    )

    metadata = {
        "model_type": "IsolationForest",
        "features": FEATURES,
        "training_samples": len(X),
        "contamination": 0.05,
        "random_state": 42,
        "dataset": "beas_observations_only.csv"
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "beas_ph_do_anomaly_metadata.joblib"
    )

    results.to_csv(
        RESULT_DIR / "anomaly_detection_results.csv",
        index=False
    )

    return results


def main():
    print("=" * 60)
    print("BEAS RIVER ANOMALY DETECTION")
    print("=" * 60)

    data = prepare_data()

    print("Paired observations:", len(data))
    print("Stations:", data["station_name"].nunique())
    print("Features:", FEATURES)

    results = train_detector(data)

    print("\nTRAINING RESULTS")
    print("Normal observations:",
          (results["anomaly_prediction"] == 1).sum())

    print("Flagged observations:",
          (results["anomaly_prediction"] == -1).sum())

    print("\nSaved model files in:", MODEL_DIR)
    print("Saved results in:", RESULT_DIR)


if __name__ == "__main__":
    main()

