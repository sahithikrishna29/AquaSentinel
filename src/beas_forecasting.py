import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_FILE = "RiverWaterQualityOverYears_PreprocessedData.csv"
OUTPUT_DIR = "Results/beas_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDICATORS = {
    "DO_min": "Dissolved Oxygen (mg/L) Min",
    "pH_average": None,
    "BOD_max": "BOD (mg/L) Max",
    "Nitrate_max": "Nitrate N (mg/L) Max"
}

def main():
    print("=" * 60)
    print("BEAS RIVER - ANNUAL WATER QUALITY FORECASTING")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    beas = df[
        df["Monitoring Location"].astype(str).str.contains(
            "BEAS", case=False, na=False
        )
    ].copy()

    beas["Year"] = pd.to_numeric(beas["Year"], errors="coerce")

    beas["pH_average"] = (
        pd.to_numeric(beas["pH Min"], errors="coerce")
        + pd.to_numeric(beas["pH Max"], errors="coerce")
    ) / 2

    for name, column in INDICATORS.items():
        if column is not None:
            beas[name] = pd.to_numeric(beas[column], errors="coerce")

    annual = beas.groupby("Year")[list(INDICATORS.keys())].mean()
    annual = annual.dropna().sort_index()

    print("\nAVAILABLE ANNUAL DATA:")
    print(annual.round(3).to_string())

    if len(annual) < 5:
        raise ValueError("Not enough annual observations for forecasting.")

    results = []
    forecast_rows = []

    for name in INDICATORS:
        series = annual[name].dropna()
        years = series.index.to_numpy(dtype=float)
        values = series.to_numpy(dtype=float)

        # Hold out the last two available years for a basic evaluation.
        train_years = years[:-2]
        train_values = values[:-2]
        test_years = years[-2:]
        test_values = values[-2:]

        evaluation_model = LinearRegression()
        evaluation_model.fit(train_years.reshape(-1, 1), train_values)

        test_predictions = evaluation_model.predict(
            test_years.reshape(-1, 1)
        )

        mae = mean_absolute_error(test_values, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_values, test_predictions))

        results.append({
            "indicator": name,
            "training_years": len(train_years),
            "testing_years": len(test_years),
            "MAE": mae,
            "RMSE": rmse
        })

        # Fit using all available annual observations for exploratory forecasts.
        final_model = LinearRegression()
        final_model.fit(years.reshape(-1, 1), values)

        future_years = np.array([2024, 2025, 2026], dtype=float)
        future_values = final_model.predict(
            future_years.reshape(-1, 1)
        )

        for year, prediction in zip(future_years, future_values):
            forecast_rows.append({
                "indicator": name,
                "year": int(year),
                "forecast_value": float(prediction)
            })

        plt.figure(figsize=(9, 5))
        plt.plot(years, values, marker="o", label="Historical annual average")
        plt.plot(
            future_years, future_values,
            marker="o", linestyle="--", label="Exploratory forecast"
        )
        plt.axvline(2023, linestyle=":", label="Forecast starts")
        plt.title(f"Beas River - {name} Annual Trend and Forecast")
        plt.xlabel("Year")
        plt.ylabel("Annual average")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{name}_forecast.png"),
            dpi=200
        )
        plt.close()

        print(f"\n{name}")
        print(f"Holdout MAE: {mae:.4f}")
        print(f"Holdout RMSE: {rmse:.4f}")

    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUT_DIR, "forecast_evaluation.csv"),
        index=False
    )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv(
        os.path.join(OUTPUT_DIR, "forecast_2024_2026.csv"),
        index=False
    )

    annual.to_csv(
        os.path.join(OUTPUT_DIR, "annual_historical_averages.csv")
    )

    print("\nFORECAST VALUES:")
    print(forecast_df.round(3).to_string(index=False))

    print("\nFILES SAVED IN:", OUTPUT_DIR)
    print("ANNUAL FORECASTING COMPLETED.")

if __name__ == "__main__":
    main()
