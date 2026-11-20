import os
import warnings
import joblib
import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

DATA_FILE = "RiverWaterQualityOverYears_PreprocessedData.csv"
OUTPUT_DIR = "Results/beas_explainable_ai"
MODEL_DIR = "models/member3"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Temperature (°C) Min",
    "Temperature (°C) Max",
    "Dissolved Oxygen (mg/L) Min",
    "Dissolved Oxygen (mg/L) Max",
    "pH Min",
    "pH Max",
    "Conductivity (µmho/cm) Min",
    "Conductivity (µmho/cm) Max",
    "BOD (mg/L) Min",
    "BOD (mg/L) Max",
    "Nitrate N (mg/L) Min",
    "Nitrate N (mg/L) Max"
]

def main():
    print("=" * 60)
    print("BEAS RIVER - SHAP EXPLAINABILITY")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)
    df = df[
        df["Monitoring Location"].astype(str).str.contains(
            "BEAS", case=False, na=False
        )
    ].copy()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Contaminated"], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    print("Beas records:", len(X))
    print("Features:", len(FEATURES))
    print("Class counts:")
    print(y.value_counts().to_string())

    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Not enough records in each class for 5-fold stratified CV.")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nRunning 5-fold cross-validation...")
    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "records": len(X),
        "positive_records": int((y == 1).sum()),
        "negative_records": int((y == 0).sum()),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1_score": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities)
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "cross_validation_metrics.csv"),
        index=False
    )

    with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
        f.write("BEAS CONTAMINATION MODEL - 5-FOLD CROSS-VALIDATION\n")
        f.write("=" * 55 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y, predictions, zero_division=0))
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y, predictions)))

    print("\nCROSS-VALIDATION RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nFitting final model for SHAP explanations...")
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODEL_DIR, "beas_contamination_rf.joblib"))

    explainer = shap.TreeExplainer(model)
    shap_result = explainer(X)
    shap_values = shap_result.values

    # Select SHAP values for the contaminated class (class 1).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap": abs(shap_values).mean(axis=0)
    }).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(
        os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
        index=False
    )

    shap.summary_plot(
        shap_values, X, show=False, max_display=len(FEATURES)
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_summary.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close()

    shap.summary_plot(
        shap_values, X, plot_type="bar", show=False,
        max_display=len(FEATURES)
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close()

    print("\nTOP SHAP FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nSHAP OUTPUTS SAVED IN:", OUTPUT_DIR)
    print("MODEL SAVED IN:", MODEL_DIR)
    print("\nSHAP EXPLAINABILITY COMPLETED.")

if __name__ == "__main__":
    main()
