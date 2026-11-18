import pandas as pd

file = "data/processed/beas_integrated_long.csv"
df = pd.read_csv(file)

print("MONTHLY OBSERVATION AVAILABILITY")
print("=" * 50)

monthly = df[df["record_type"] == "monthly_observation"].copy()
monthly["value"] = pd.to_numeric(monthly["value"], errors="coerce")

print(
    monthly.groupby("parameter")
    .agg(
        total_records=("value", "size"),
        actual_values=("value", "count"),
        missing_values=("value", lambda x: x.isna().sum()),
        stations=("station_name", "nunique")
    )
    .to_string()
)

print("\nANNUAL RANGE AVAILABILITY")
print("=" * 50)

annual = df[df["record_type"] == "annual_min_max_range"].copy()

for col in ["min_value", "max_value"]:
    annual[col] = pd.to_numeric(annual[col], errors="coerce")

print(
    annual.groupby("parameter")
    .agg(
        total_records=("parameter", "size"),
        valid_min=("min_value", "count"),
        valid_max=("max_value", "count"),
        stations=("station_name", "nunique")
    )
    .to_string()
)
