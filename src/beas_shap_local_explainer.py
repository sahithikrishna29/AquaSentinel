
import os
import warnings
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "RiverWaterQualityOverYears_PreprocessedData.csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "member3"
    / "beas_contamination_rf.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "Results" / "beas_shap_advanced"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# FEATURES USED DURING ORIGINAL MODEL TRAINING
# --------------------------------------------------

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

# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def load_beas_data():

    df = pd.read_csv(DATA_FILE)

    df["source_row_index"] = df.index

    df = df[
        df["Monitoring Location"]
        .astype(str)
        .str.contains("BEAS", case=False, na=False)
    ].copy()

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = pd.to_numeric(
        df["Contaminated"],
        errors="coerce"
    )

    valid = X.notna().all(axis=1) & y.isin([0, 1])

    X = X.loc[valid].reset_index(drop=True)

    y = y.loc[valid].astype(int).reset_index(drop=True)

    source_indices = (
        df.loc[valid, "source_row_index"]
        .reset_index(drop=True)
    )

    if len(X) == 0:
        raise ValueError("No valid Beas River observations found.")

    if y.nunique() != 2:
        raise ValueError("Both contamination classes are required.")

    return X, y, source_indices


# --------------------------------------------------
# GENERATE SHAP EXPLANATIONS
# --------------------------------------------------

def main():

    print("=" * 65)
    print("BEAS RIVER - ADVANCED LOCAL SHAP EXPLAINABILITY")
    print("=" * 65)

    print("\nLoading trained model...")

    model = joblib.load(MODEL_FILE)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model does not contain feature metadata.")

    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Dataset features do not match the trained model."
        )

    print("Model loaded:", type(model).__name__)

    print("\nLoading Beas River dataset...")

    X, y, source_indices = load_beas_data()

    print("Valid observations:", len(X))
    print("Features:", len(FEATURES))

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    class_one_index = list(model.classes_).index(1)

    contamination_probabilities = probabilities[:, class_one_index]

    # Select the most confidently predicted actual example
    # from each actual class.

    positive_candidates = np.where(y.to_numpy() == 1)[0]

    negative_candidates = np.where(y.to_numpy() == 0)[0]

    positive_index = positive_candidates[
        np.argmax(contamination_probabilities[positive_candidates])
    ]

    negative_index = negative_candidates[
        np.argmin(contamination_probabilities[negative_candidates])
    ]

    selected_indices = {
        "contaminated": int(positive_index),
        "non_contaminated": int(negative_index)
    }

    # --------------------------------------------------
    # SHAP TREE EXPLAINER
    # --------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)

    raw_values = explanation.values

    if isinstance(raw_values, list):

        shap_class_values = np.asarray(
            raw_values[class_one_index]
        )

    else:

        raw_values = np.asarray(raw_values)

        if raw_values.ndim == 3:

            shap_class_values = raw_values[
                :, :, class_one_index
            ]

        elif raw_values.ndim == 2:

            shap_class_values = raw_values

        else:

            raise ValueError(
                f"Unexpected SHAP value shape: {raw_values.shape}"
            )

    # --------------------------------------------------
    # SELECT BASE VALUES FOR CONTAMINATED CLASS
    # --------------------------------------------------

    base_values = np.asarray(explanation.base_values)

    if base_values.ndim == 2:

        class_base_values = base_values[:, class_one_index]

    elif base_values.ndim == 1:

        if len(base_values) == len(X):

            class_base_values = base_values

        elif len(base_values) == len(model.classes_):

            class_base_values = np.full(
                len(X),
                base_values[class_one_index]
            )

        else:

            raise ValueError("Unexpected SHAP base-value shape.")

    elif base_values.ndim == 0:

        class_base_values = np.full(
            len(X),
            float(base_values)
        )

    else:

        raise ValueError("Unsupported SHAP base-value format.")

    # --------------------------------------------------
    # GENERATE INDIVIDUAL EXPLANATIONS
    # --------------------------------------------------

    contribution_records = []

    for observation_type, row_index in selected_indices.items():

        print(f"\nExplaining: {observation_type}")

        local_explanation = shap.Explanation(
            values=shap_class_values[row_index],
            base_values=class_base_values[row_index],
            data=X.iloc[row_index].to_numpy(),
            feature_names=FEATURES
        )

        waterfall_file = (
            OUTPUT_DIR
            / f"waterfall_{observation_type}.png"
        )

        shap.plots.waterfall(
            local_explanation,
            max_display=len(FEATURES),
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            waterfall_file,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        # Save individual feature contributions.

        for feature_index, feature in enumerate(FEATURES):

            shap_value = float(
                shap_class_values[row_index, feature_index]
            )

            if shap_value > 0:
                impact = "Increases contamination output"

            elif shap_value < 0:
                impact = "Decreases contamination output"

            else:
                impact = "No contribution"

            contribution_records.append({

                "observation_type": observation_type,

                "source_row_index": int(
                    source_indices.iloc[row_index]
                ),

                "actual_class": int(y.iloc[row_index]),

                "predicted_class": int(
                    predictions[row_index]
                ),

                "contamination_probability": round(
                    float(
                        contamination_probabilities[row_index]
                    ),
                    6
                ),

                "feature": feature,

                "feature_value": float(
                    X.iloc[row_index, feature_index]
                ),

                "shap_value_class_1": shap_value,

                "impact_direction": impact

            })

    # --------------------------------------------------
    # SAVE CONTRIBUTION REPORT
    # --------------------------------------------------

    contributions = pd.DataFrame(contribution_records)

    contributions = contributions.sort_values(
        ["observation_type", "shap_value_class_1"],
        ascending=[True, False]
    )

    contributions.to_csv(
        OUTPUT_DIR / "local_shap_contributions.csv",
        index=False
    )

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    summary = pd.DataFrame({

        "observation_type": list(selected_indices.keys()),

        "source_row_index": [
            int(source_indices.iloc[index])
            for index in selected_indices.values()
        ],

        "actual_class": [
            int(y.iloc[index])
            for index in selected_indices.values()
        ],

        "predicted_class": [
            int(predictions[index])
            for index in selected_indices.values()
        ],

        "contamination_probability": [
            float(contamination_probabilities[index])
            for index in selected_indices.values()
        ]

    })

    summary.to_csv(
        OUTPUT_DIR / "local_explanation_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # DOCUMENT METHODOLOGY
    # --------------------------------------------------

    with open(
        OUTPUT_DIR / "methodology.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BEAS RIVER - ADVANCED LOCAL SHAP EXPLAINABILITY\n"
        )

        file.write("=" * 60 + "\n")

        file.write(f"Valid Beas observations: {len(X)}\n")

        file.write(f"Features: {len(FEATURES)}\n")

        file.write(
            f"Non-contaminated observations: {(y == 0).sum()}\n"
        )

        file.write(
            f"Contaminated observations: {(y == 1).sum()}\n"
        )

        file.write(
            "Model: Existing trained RandomForestClassifier\n"
        )

        file.write(
            "Explainer: SHAP TreeExplainer\n"
        )

        file.write(
            "Explained output: Class 1 (contaminated)\n"
        )

        file.write(
            "\nIMPORTANT METHODOLOGICAL LIMITATION:\n"
        )

        file.write(
            "The existing model was fitted on the Beas observations "
            "by the original training pipeline. These local explanations "
            "are explanations of that fitted model on its training "
            "observations, not independent test-set explanations.\n"
        )

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\nLOCAL EXPLANATION SUMMARY")

    print(summary.to_string(index=False))

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):

        if file.is_file():

            print(file.name)

    print("\nAdvanced local SHAP analysis completed.")


if __name__ == "__main__":

    main()
