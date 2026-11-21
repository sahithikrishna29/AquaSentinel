
import streamlit as st

st.set_page_config(
    page_title="AquaSentinel Dashboard",
    page_icon="💧",
    layout="wide"
)

st.title("💧 AquaSentinel")
st.subheader("Intelligent River Water Quality Monitoring System")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Water Quality Prediction",
        "Anomaly Detection",
        "Forecasting",
        "Sensor Health",
        "SHAP Explainability",
        "Pollution Alerts"
    ]
)

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "AquaSentinel is an intelligent river water quality "
        "monitoring system using machine learning."
    )

    st.info("Select a module from the sidebar to continue.")


elif page == "Water Quality Prediction":
    st.header("Water Quality Prediction")
    st.write("Enter water quality sensor readings.")

    with st.form("prediction_form"):
        ph = st.number_input(
            "pH", min_value=0.0, max_value=14.0, value=7.0
        )
        hardness = st.number_input(
            "Hardness (mg/L)", min_value=0.0, value=200.0
        )
        solids = st.number_input(
            "Solids (mg/L)", min_value=0.0, value=20000.0
        )
        chloramines = st.number_input(
            "Chloramines (mg/L)", min_value=0.0, value=7.0
        )
        sulfate = st.number_input(
            "Sulfate (mg/L)", min_value=0.0, value=300.0
        )
        conductivity = st.number_input(
            "Conductivity", min_value=0.0, value=400.0
        )
        organic_carbon = st.number_input(
            "Organic Carbon (mg/L)", min_value=0.0, value=14.0
        )
        trihalomethanes = st.number_input(
            "Trihalomethanes", min_value=0.0, value=60.0
        )
        turbidity = st.number_input(
            "Turbidity (NTU)", min_value=0.0, value=4.0
        )

        submitted = st.form_submit_button(
            "Predict Water Quality"
        )

    if submitted:
        import requests

        payload = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            st.subheader("Prediction Results")

            st.write("Classification:")
            st.json(result.get("classification", {}))

            st.write("Anomaly Detection:")
            st.json(result.get("anomaly_detection", {}))

        except requests.exceptions.RequestException as e:
            st.error(f"API connection error: {e}")
            st.info(
                "Make sure the FastAPI server is running "
                "on port 8000."
            )

elif page == "Anomaly Detection":
    st.header("Beas River Anomaly Detection")
    st.write("Historical anomaly detection results")

    import pandas as pd
    from pathlib import Path

    file_path = Path(
        "Results/beas_anomaly_detection/anomaly_detection_results.csv"
    )

    if file_path.exists():
        try:
            df = pd.read_csv(file_path)

            st.success("Anomaly detection results loaded!")

            st.subheader("Dataset Overview")
            st.metric("Total Records", len(df))

            st.subheader("Anomaly Detection Data")
            st.dataframe(df, use_container_width=True)

            st.subheader("Available Columns")
            st.write(df.columns.tolist())

            # Anomaly Detection Statistics
            st.subheader("Anomaly Detection Statistics")

            status = df["anomaly_status"].astype(str).str.strip().str.upper()
            normal_count = (status == "NORMAL").sum()
            anomaly_count = (status == "ANOMALY").sum()
            unknown_count = len(df) - normal_count - anomaly_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Observations", len(df))
            col2.metric("Normal Readings", int(normal_count))
            col3.metric("Anomalies Detected", int(anomaly_count))

            if len(df) > 0:
                anomaly_percentage = (anomaly_count / len(df)) * 100
                st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

            if unknown_count > 0:
                st.warning(
                    f"{unknown_count} records have unrecognized anomaly status."
                )

            # Anomaly Status Distribution
            st.subheader("Anomaly Status Distribution")

            status_counts = status.value_counts()
            st.bar_chart(status_counts)

            # Station-wise Anomaly Analysis
            st.subheader("Station-wise Anomaly Distribution")

            anomaly_df = df[status == "ANOMALY"]

            if not anomaly_df.empty:
                station_counts = (
                    anomaly_df.groupby("station_name")
                    .size()
                    .sort_values(ascending=False)
                )
                st.bar_chart(station_counts)
            else:
                st.info("No anomalies found in the dataset.")

            # Historical Water Quality Trends
            st.subheader("Water Quality Trends")

            trend_df = df.copy()
            trend_df["date"] = pd.to_datetime(
                trend_df["date"], errors="coerce"
            )
            trend_df = trend_df.dropna(subset=["date"])
            trend_df = trend_df.sort_values("date")

            if not trend_df.empty:
                for parameter in ["pH", "Dissolved Oxygen"]:
                    if parameter in trend_df.columns:
                        st.write(f"{parameter} Trend")
                        chart_df = trend_df[
                            ["date", parameter]
                        ].copy()
                        chart_df[parameter] = pd.to_numeric(
                            chart_df[parameter], errors="coerce"
                        )
                        chart_df = chart_df.dropna()
                        chart_df = chart_df.set_index("date")
                        st.line_chart(chart_df)

            st.download_button(
                "Download Anomaly Results",
                data=df.to_csv(index=False),
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Unable to load anomaly results: {e}")

    else:
        st.warning("Anomaly detection results file not found.")


elif page == "Forecasting":
    st.header("Beas River Water Quality Forecasting")
    st.write("Historical model forecasts for 2024–2026")

    import pandas as pd
    from pathlib import Path

    forecast_path = Path(
        "Results/beas_forecasting/forecast_2024_2026.csv"
    )
    evaluation_path = Path(
        "Results/beas_forecasting/forecast_evaluation.csv"
    )

    if forecast_path.exists():
        try:
            forecast_df = pd.read_csv(forecast_path)

            st.success("Forecasting results loaded successfully!")

            st.subheader("Forecast Dataset")
            st.dataframe(forecast_df, use_container_width=True)

            indicators = forecast_df["indicator"].dropna().unique()

            selected_indicator = st.selectbox(
                "Select Water Quality Indicator",
                indicators
            )

            selected_df = forecast_df[
                forecast_df["indicator"] == selected_indicator
            ].copy()

            selected_df["year"] = pd.to_numeric(
                selected_df["year"], errors="coerce"
            )
            selected_df["forecast_value"] = pd.to_numeric(
                selected_df["forecast_value"], errors="coerce"
            )

            selected_df = selected_df.dropna(
                subset=["year", "forecast_value"]
            ).sort_values("year")

            st.subheader(f"{selected_indicator} Forecast Trend")

            if not selected_df.empty:
                chart_df = selected_df.set_index("year")[
                    ["forecast_value"]
                ]
                st.line_chart(chart_df)

                st.metric(
                    "Final Forecast Value",
                    f"{selected_df['forecast_value'].iloc[-1]:.3f}"
                )

            st.download_button(
                "Download Forecast Results",
                data=forecast_df.to_csv(index=False),
                file_name="forecast_2024_2026.csv",
                mime="text/csv"
            )

            if evaluation_path.exists():
                st.subheader("Forecast Model Evaluation")

                evaluation_df = pd.read_csv(evaluation_path)

                st.dataframe(
                    evaluation_df,
                    use_container_width=True
                )

                st.write(
                    "MAE: Mean Absolute Error | "
                    "RMSE: Root Mean Squared Error"
                )

        except Exception as e:
            st.error(f"Unable to load forecasting results: {e}")

    else:
        st.warning("Forecasting results file not found.")






elif page == "Sensor Health":
    st.header("Beas River Sensor Health Monitoring")
    st.write("Historical monitoring data quality assessment")

    import pandas as pd
    from pathlib import Path

    file_path = Path(
        "Results/sensor_health/sensor_health_report.csv"
    )

    if file_path.exists():
        try:
            df = pd.read_csv(file_path)

            st.success("Sensor health report loaded successfully!")

            numeric_columns = [
                "total_records",
                "missing_readings",
                "invalid_readings",
                "repeated_values",
                "date_gaps_over_60_days"
            ]

            for column in numeric_columns:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                ).fillna(0)

            st.subheader("Sensor Health Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Monitoring Stations",
                df["station_name"].nunique()
            )
            col2.metric(
                "Missing Readings",
                int(df["missing_readings"].sum())
            )
            col3.metric(
                "Invalid Readings",
                int(df["invalid_readings"].sum())
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "Repeated Values",
                int(df["repeated_values"].sum())
            )
            col5.metric(
                "Date Gaps Over 60 Days",
                int(df["date_gaps_over_60_days"].sum())
            )

            st.subheader("Sensor Health Report")
            st.dataframe(df, use_container_width=True)

            st.subheader("Parameter-wise Data Quality")

            summary = df.groupby("parameter")[
                [
                    "missing_readings",
                    "invalid_readings",
                    "repeated_values"
                ]
            ].sum()

            st.bar_chart(summary)

            st.subheader("Station-wise Missing Readings")

            station_summary = df.groupby("station_name")[
                "missing_readings"
            ].sum()

            st.bar_chart(station_summary)

            st.download_button(
                "Download Sensor Health Report",
                data=df.to_csv(index=False),
                file_name="sensor_health_report.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Unable to load sensor health report: {e}")

    else:
        st.warning("Sensor health report file not found.")


elif page == "SHAP Explainability":
    st.header("AquaSentinel SHAP Explainability")
    st.write("Understanding water quality model predictions")

    import pandas as pd
    from pathlib import Path

    folders = [
        Path("Results/beas_shap_advanced"),
        Path("Results/beas_explainable_ai")
    ]

    files = [
        folder / "local_shap_contributions.csv"
        for folder in folders
    ]

    shap_path = next(
        (path for path in files if path.exists()), None
    )

    if shap_path is not None:
        try:
            df = pd.read_csv(shap_path)

            st.success("SHAP results loaded successfully!")

            st.subheader("SHAP Explanation Data")
            st.dataframe(df, use_container_width=True)

            st.subheader("Global Feature Importance")

            df["shap_value_class_1"] = pd.to_numeric(
                df["shap_value_class_1"],
                errors="coerce"
            )

            importance = (
                df.groupby("feature")["shap_value_class_1"]
                .apply(lambda x: x.abs().mean())
                .sort_values(ascending=False)
            )

            st.bar_chart(importance)

            st.caption(
                "Mean absolute SHAP contribution across "
                "the saved observations."
            )

            st.subheader("Local Prediction Explanation")

            observation_ids = df[
                "source_row_index"
            ].dropna().unique().tolist()

            selected_id = st.selectbox(
                "Select Observation",
                observation_ids
            )

            selected_df = df[
                df["source_row_index"] == selected_id
            ]

            st.dataframe(
                selected_df,
                use_container_width=True
            )

            st.subheader("Feature Contributions")

            contributions = (
                selected_df.groupby("feature")[
                    "shap_value_class_1"
                ].sum()
            )

            st.bar_chart(contributions)

            st.info(
                "Positive SHAP values contribute toward "
                "the model's class 1 output; negative values "
                "contribute in the opposite direction."
            )

            st.download_button(
                "Download SHAP Results",
                data=df.to_csv(index=False),
                file_name="local_shap_contributions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Unable to load SHAP results: {e}")

    else:
        st.warning("Local SHAP results file not found.")





elif page == "Pollution Alerts":
    st.header("Pollution Alerts")
    st.write("Threshold-based water quality monitoring")

    from src.anomaly_detector import anomaly_detector

    st.subheader("Enter Water Quality Parameters")

    ph = st.number_input("pH", value=7.0)
    turbidity = st.number_input("Turbidity (NTU)", value=4.0)
    do = st.number_input("Dissolved Oxygen (mg/L)", value=8.0)
    conductivity = st.number_input(
        "Conductivity (uS/cm)", value=400.0
    )

    if st.button("Check Pollution Alerts"):
        readings = {
            "ph": ph,
            "Turbidity": turbidity,
            "do": do,
            "Conductivity": conductivity
        }

        statuses = anomaly_detector._check_individual_params(
            readings
        )

        st.subheader("Water Quality Alert Results")

        for parameter, status in statuses.items():
            if status == "Normal":
                st.success(f"{parameter}: {status}")
            elif status.startswith(("Low", "High")):
                st.warning(f"{parameter}: {status}")
            else:
                st.info(f"{parameter}: {status}")

        st.caption(
            "Alerts use project-configured thresholds, "
            "not independently verified regulatory limits."
        )
