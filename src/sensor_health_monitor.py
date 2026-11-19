
"""
AquaSentinel - Beas River Sensor Health Monitoring
Member 2

Checks historical observation records for:
1. Missing readings
2. Invalid pH readings
3. Invalid dissolved oxygen readings
4. Repeated readings
5. Observation date gaps

Note: This evaluates historical data quality.
It does not establish live sensor connectivity.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "beas_observations_only.csv"

OUTPUT_DIR = ROOT / "Results" / "sensor_health"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETERS = ["pH", "Dissolved Oxygen"]

# Broad physical plausibility limits, not regulatory compliance limits.
LIMITS = {
    "pH": (0, 14),
    "Dissolved Oxygen": (0, 20)
}


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[
        df["parameter"].isin(PARAMETERS)
        & df["record_type"].isin([
            "sample_observation",
            "monthly_observation"
        ])
    ].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def check_health(df):
    results = []

    for parameter in PARAMETERS:
        part = df[df["parameter"] == parameter].copy()

        for station, group in part.groupby("station_name"):
            group = group.sort_values("date")

            missing_count = int(group["value"].isna().sum())

            low, high = LIMITS[parameter]

            invalid_count = int(
                (
                    group["value"].notna()
                    & (
                        (group["value"] < low)
                        | (group["value"] > high)
                    )
                ).sum()
            )

            valid_values = group["value"].dropna()

            repeated_count = int(
                valid_values.duplicated(keep=False).sum()
            )

            date_gaps = group["date"].dropna().sort_values().diff()
            large_gaps = int((date_gaps.dt.days > 60).sum())

            results.append({
                "station_name": station,
                "parameter": parameter,
                "total_records": len(group),
                "missing_readings": missing_count,
                "invalid_readings": invalid_count,
                "repeated_values": repeated_count,
                "date_gaps_over_60_days": large_gaps
            })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("BEAS RIVER SENSOR HEALTH REPORT")
    print("=" * 60)

    df = load_data()

    print("Total historical records:", len(df))
    print("Monitoring stations:", df["station_name"].nunique())

    report = check_health(df)

    output_file = OUTPUT_DIR / "sensor_health_report.csv"
    report.to_csv(output_file, index=False)

    print("\nSENSOR HEALTH SUMMARY")
    print(report[
        [
            "parameter",
            "total_records",
            "missing_readings",
            "invalid_readings",
            "repeated_values",
            "date_gaps_over_60_days"
        ]
    ].groupby("parameter").sum().to_string())

    print("\nStation-level report saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
