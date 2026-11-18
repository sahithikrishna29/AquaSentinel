from pathlib import Path
import pandas as pd

# --------------------------------------------------
# AQUASENTINEL - BEAS RIVER DATA INTEGRATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "beas_real_source_compilation_long.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read original source compilation
df = pd.read_csv(INPUT_FILE)

# Validate required columns
required_columns = [
    "record_type",
    "station_name",
    "parameter",
    "unit",
    "source_agency",
    "source_document",
    "source_url",
    "synthetic",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Preserve original parameter names
df["original_parameter"] = df["parameter"]

# Standardize equivalent parameter names
parameter_mapping = {
    "DO": "Dissolved Oxygen",
    "BOD": "Biochemical Oxygen Demand",
}

df["parameter"] = (
    df["parameter"].replace(parameter_mapping)
)

# Check record types
allowed_types = {
    "annual_min_max_range",
    "monthly_observation",
}

unexpected_types = set(df["record_type"].dropna()) - allowed_types

if unexpected_types:
    raise ValueError(
        f"Unexpected record types: {unexpected_types}"
    )

# Keep source records separate
monthly_df = df[
    df["record_type"] == "monthly_observation"
].copy()

annual_df = df[
    df["record_type"] == "annual_min_max_range"
].copy()

# Do not create artificial measurements
# Do not fill missing values
# Do not convert BDL to zero
# Do not merge different station names automatically

# Save processed datasets
df.to_csv(
    OUTPUT_DIR / "beas_integrated_long.csv",
    index=False
)

monthly_df.to_csv(
    OUTPUT_DIR / "beas_monthly_observations.csv",
    index=False
)

annual_df.to_csv(
    OUTPUT_DIR / "beas_annual_ranges.csv",
    index=False
)

# Generate a parameter summary
summary = (
    df.groupby(
        ["record_type", "parameter", "unit"],
        dropna=False
    )
    .size()
    .reset_index(name="record_count")
)

summary.to_csv(
    OUTPUT_DIR / "beas_parameter_summary.csv",
    index=False
)

print("\nAquaSentinel data integration completed!")
print("----------------------------------------")
print("Original records:", len(df))
print("Monthly observations:", len(monthly_df))
print("Annual range records:", len(annual_df))
print("Unique stations:", df["station_name"].nunique())
print("Standardized parameters:", df["parameter"].nunique())
print("Synthetic records:", int(df["synthetic"].astype(str).str.lower().eq("true").sum()))

print("\nCreated files:")
for file in sorted(OUTPUT_DIR.glob("beas_*.csv")):
    print(file.name, "-", file.stat().st_size, "bytes")
