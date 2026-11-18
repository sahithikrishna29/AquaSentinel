import pandas as pd
from pathlib import Path

# Project paths
input_file = Path("data/raw/beas_master_water_quality_long.csv")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Load original master dataset
df = pd.read_csv(input_file)

# Separate annual records
annual = df[df["record_type"] == "annual_min_max_range"].copy()
other = df[df["record_type"] != "annual_min_max_range"].copy()

# Identify annual range categories
invalid = annual[
    annual["min_value"].notna()
    & annual["max_value"].notna()
    & (annual["min_value"] > annual["max_value"])
].copy()

incomplete = annual[
    annual["min_value"].isna()
    | annual["max_value"].isna()
].copy()

valid = annual[
    annual["min_value"].notna()
    & annual["max_value"].notna()
    & (annual["min_value"] <= annual["max_value"])
].copy()

# Keep all non-annual observations and valid annual ranges
cleaned = pd.concat([other, valid], ignore_index=True)

# Save separate files
cleaned.to_csv(output_dir / "beas_master_validated_long.csv", index=False)
invalid.to_csv(output_dir / "beas_invalid_annual_ranges.csv", index=False)
incomplete.to_csv(output_dir / "beas_incomplete_annual_ranges.csv", index=False)

print("DATA PREPARATION COMPLETED")
print("Original records:", len(df))
print("Validated dataset records:", len(cleaned))
print("Valid annual ranges:", len(valid))
print("Invalid annual ranges:", len(invalid))
print("Incomplete annual ranges:", len(incomplete))
print("Other observations preserved:", len(other))
print("Output folder:", output_dir)
