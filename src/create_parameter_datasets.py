import pandas as pd
from pathlib import Path

source = Path("data/processed/beas_observations_only.csv")
output_dir = Path("data/processed/parameter_wise")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

for parameter in sorted(df["parameter"].dropna().unique()):
    safe_name = (
        parameter.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    subset = df[df["parameter"] == parameter].copy()
    subset.to_csv(
        output_dir / f"{safe_name}_observations.csv",
        index=False
    )

    print(f"{parameter}: {len(subset)} records")

print("\nParameter-wise files created in:", output_dir)
