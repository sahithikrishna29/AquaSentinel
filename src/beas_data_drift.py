import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DATA_FILE = "RiverWaterQualityOverYears_PreprocessedData.csv"
OUTPUT_DIR = "Results/beas_data_drift"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Use reference-data quantiles to define comparison bins.
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 0.0001
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    ))

def main():
    print("=" * 65)
    print("BEAS RIVER - DATA DRIFT DETECTION")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)

    beas = df[
        df["Monitoring Location"].astype(str).str.contains(
            "BEAS", case=False, na=False
        )
    ].copy()

    beas["Year"] = pd.to_numeric(beas["Year"], errors="coerce")

    # Compare earlier records with later records.
    reference = beas[beas["Year"] <= 2018]
    current = beas[beas["Year"] >= 2019]

    print("\nREFERENCE PERIOD: 2012-2018")
    print("Reference records:", len(reference))

    print("\nCURRENT PERIOD: 2019-2023")
    print("Current records:", len(current))

    if len(reference) < 20 or len(current) < 20:
        raise ValueError("Not enough records for drift comparison.")

    results = []

    for feature in FEATURES:
        ref = pd.to_numeric(
            reference[feature], errors="coerce"
        ).dropna()

        cur = pd.to_numeric(
            current[feature], errors="coerce"
        ).dropna()

        if len(ref) < 5 or len(cur) < 5:
            continue

        ks_stat, p_value = ks_2samp(ref, cur)
        psi = calculate_psi(ref, cur)

        if p_value < 0.05:
            ks_result = "Significant distribution change"
        else:
            ks_result = "No statistically significant change"

        if pd.isna(psi):
            psi_result = "Not available"
        elif psi < 0.10:
            psi_result = "Low drift"
        elif psi < 0.25:
            psi_result = "Moderate drift"
        else:
            psi_result = "High drift"

        results.append({
            "feature": feature,
            "reference_count": len(ref),
            "current_count": len(cur),
            "reference_mean": ref.mean(),
            "current_mean": cur.mean(),
            "ks_statistic": ks_stat,
            "ks_p_value": p_value,
            "ks_result": ks_result,
            "psi": psi,
            "psi_result": psi_result
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "data_drift_report.csv"),
        index=False
    )

    print("\nDATA DRIFT RESULTS:")
    print(results_df.round(4).to_string(index=False))

    print("\nSUMMARY:")
    print(
        "Features with significant KS changes:",
        (results_df["ks_p_value"] < 0.05).sum()
    )
    print(
        "Features with moderate/high PSI:",
        results_df["psi_result"].isin(
            ["Moderate drift", "High drift"]
        ).sum()
    )

    print("\nREPORT SAVED IN:", OUTPUT_DIR)
    print("DATA DRIFT DETECTION COMPLETED.")

if __name__ == "__main__":
    main()
