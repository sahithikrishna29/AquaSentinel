import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = Path("data/processed/beas_observations_only.csv")
OUTPUT_DIR = Path("Results/beas_anomaly_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load observations
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "record_id", "record_type", "parameter", "value",
    "date", "date_precision", "station_name",
    "source_agency", "unit"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_columns].copy()
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Keep only pH and DO for this first analysis
df = df[
    df["parameter"].isin(["pH", "Dissolved Oxygen"])
].dropna(subset=["value", "station_name", "source_agency"])

# --------------------------------------------------
# 3. Calculate station-wise statistics
# --------------------------------------------------

group_columns = [
    "source_agency",
    "record_type",
    "parameter",
    "station_name"
]

summary = (
    df.groupby(group_columns)["value"]
    .agg(
        observation_count="count",
        minimum="min",
        maximum="max",
        average="mean",
        median="median",
        standard_deviation="std"
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "station_parameter_summary.csv",
    index=False
)

# --------------------------------------------------
# 4. Exploratory IQR anomaly detection
# --------------------------------------------------

# This is a statistical screening method, not a
# confirmed pollution or regulatory classification.

flagged_groups = []

for group_key, group in df.groupby(group_columns):
    group = group.copy()

    # Require at least 8 observations for screening
    if len(group) < 8:
        continue

    q1 = group["value"].quantile(0.25)
    q3 = group["value"].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    group["lower_iqr_limit"] = lower_limit
    group["upper_iqr_limit"] = upper_limit

    group["statistical_flag"] = (
        (group["value"] < lower_limit) |
        (group["value"] > upper_limit)
    )

    group["flag_reason"] = group["statistical_flag"].map({
        True: "Outside station-wise IQR screening limits",
        False: "Within station-wise IQR screening limits"
    })

    flagged_groups.append(group)

if flagged_groups:
    results = pd.concat(flagged_groups, ignore_index=True)
else:
    results = pd.DataFrame()

# --------------------------------------------------
# 5. Save all screening results
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "anomaly_screening_results.csv",
    index=False
)

if not results.empty:
    anomalies = results[
        results["statistical_flag"]
    ].copy()
else:
    anomalies = results.copy()

anomalies.to_csv(
    OUTPUT_DIR / "flagged_observations.csv",
    index=False
)

# --------------------------------------------------
# 6. Print report
# --------------------------------------------------

print("\nBEAS ANOMALY SCREENING COMPLETED")
print("----------------------------------")
print("Input observations:", len(df))
print("Station groups:", len(summary))
print("Statistically flagged observations:", len(anomalies))
print("\nSource-wise results:")

if not results.empty:
    print(
        results.groupby(
            ["source_agency", "parameter"]
        )["statistical_flag"]
        .agg(["count", "sum"])
        .to_string()
    )

print("\nOutput folder:", OUTPUT_DIR)
print("1. station_parameter_summary.csv")
print("2. anomaly_screening_results.csv")
print("3. flagged_observations.csv")

print("\nNOTE: Statistical flags are not confirmed pollution events.")
