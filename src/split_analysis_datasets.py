import pandas as pd
from pathlib import Path

input_file = Path("data/processed/beas_master_validated_long.csv")
output_dir = Path("data/processed")

df = pd.read_csv(input_file)

observations = df[
    df["record_type"].isin([
        "sample_observation",
        "monthly_observation"
    ])
].copy()

annual = df[
    df["record_type"] == "annual_min_max_range"
].copy()

observations.to_csv(
    output_dir / "beas_observations_only.csv",
    index=False
)

annual.to_csv(
    output_dir / "beas_annual_ranges_valid.csv",
    index=False
)

print("ANALYSIS DATASETS CREATED")
print("Individual and monthly observations:", len(observations))
print("Valid annual ranges:", len(annual))
print("Total:", len(observations) + len(annual))
