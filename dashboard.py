
import streamlit as st
import pandas as pd
import requests
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AquaSentinel Dashboard",
    page_icon="💧",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Updated API URL: Beas-integrated FastAPI server
API_BASE_URL = "http://127.0.0.1:8002"

RESULTS_DIR = BASE_DIR / "Results"

ANOMALY_FILE = (
    RESULTS_DIR / "beas_anomaly_detection"
    / "anomaly_detection_results.csv"
)

FORECAST_FILE = (
    RESULTS_DIR / "beas_forecasting"
    / "forecast_2024_2026.csv"
)

EVALUATION_FILE = (
    RESULTS_DIR / "beas_forecasting"
    / "forecast_evaluation.csv"
)

SENSOR_FILE = (
    RESULTS_DIR / "sensor_health"
    / "sensor_health_report.csv"
)

SHAP_FILES = [
    RESULTS_DIR / "beas_shap_advanced"
    / "local_shap_contributions.csv",
    RESULTS_DIR / "beas_explainable_ai"
    / "local_shap_contributions.csv"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(file_path):
    """Load a CSV file safely."""

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.error(
            f"Unable to read {file_path.name}: {error}"
        )
        return None


def check_api_health():
    """Check whether FastAPI is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException:
        return None


def display_api_status():
    """Display backend connection status."""

    health = check_api_health()

    if health is None:

        st.sidebar.error("FastAPI: Disconnected")

        st.sidebar.caption(
            "Start the FastAPI server on port 8002."
        )

        return

    st.sidebar.success("FastAPI: Connected")

    models = health.get("models", {})

    st.sidebar.write(
        "Classifier:",
        "Loaded" if models.get("classifier")
        else "Unavailable"
    )

    st.sidebar.write(
        "Beas anomaly detector:",
        "Loaded" if models.get("anomaly_detector")
        else "Unavailable"
    )


# ============================================================
# HEADER AND NAVIGATION
# ============================================================

st.title("💧 AquaSentinel")

st.subheader(
    "Intelligent River Water Quality Monitoring System"
)

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

display_api_status()


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        AquaSentinel is an intelligent river water quality
        monitoring system that combines machine learning,
        anomaly detection, forecasting and explainable AI.

        The dashboard provides a unified interface for
        monitoring water quality and identifying unusual
        environmental observations.
        """
    )

    st.info(
        "Select a module from the sidebar to continue."
    )

    st.subheader("Available Modules")

    col1, col2, col3 = st.columns(3)

    col1.metric("Core ML Tasks", "2")
    col2.metric("Monitoring Modules", "3")
    col3.metric("Explainability Modules", "2")

    st.markdown(
        """
        **System capabilities**

        - Water quality classification
        - Beas River pH and Dissolved Oxygen anomaly detection
        - Water quality forecasting results
        - Sensor health monitoring
        - SHAP explainability
        - Threshold-based pollution alerts
        """
    )


# ============================================================
# 2. WATER QUALITY PREDICTION
# ============================================================

elif page == "Water Quality Prediction":

    st.header("Water Quality Prediction")

    st.write(
        "Enter sensor readings to perform water quality "
        "classification and Beas anomaly detection."
    )

    with st.form("prediction_form"):

        st.subheader("Water Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:

            ph = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=7.0
            )

            dissolved_oxygen = st.number_input(
                "Dissolved Oxygen (mg/L)",
                min_value=0.0,
                value=6.8
            )

            hardness = st.number_input(
                "Hardness (mg/L)",
                min_value=0.0,
                value=200.0
            )

            solids = st.number_input(
                "Solids (mg/L)",
                min_value=0.0,
                value=20000.0
            )

            chloramines = st.number_input(
                "Chloramines (mg/L)",
                min_value=0.0,
                value=7.0
            )

            sulfate = st.number_input(
                "Sulfate (mg/L)",
                min_value=0.0,
                value=300.0
            )

        with col2:

            conductivity = st.number_input(
                "Conductivity",
                min_value=0.0,
                value=400.0
            )

            organic_carbon = st.number_input(
                "Organic Carbon (mg/L)",
                min_value=0.0,
                value=14.0
            )

            trihalomethanes = st.number_input(
                "Trihalomethanes",
                min_value=0.0,
                value=60.0
            )

            turbidity = st.number_input(
                "Turbidity (NTU)",
                min_value=0.0,
                value=4.0
            )

            temperature = st.number_input(
                "Temperature",
                value=28.3
            )

            tds = st.number_input(
                "Total Dissolved Solids (TDS)",
                min_value=0.0,
                value=320.0
            )

        submitted = st.form_submit_button(
            "Predict Water Quality",
            use_container_width=True
        )

    if submitted:

        payload = {
            "ph": ph,
            "do": dissolved_oxygen,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity,
            "temperature": temperature,
            "tds": tds
        }

        try:

            with st.spinner("Analyzing water quality..."):

                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()

                result = response.json()

            st.success(
                "Prediction completed successfully!"
            )

            st.subheader("Water Quality Classification")

            classification = result.get(
                "classification", {}
            )

            quality_class = classification.get(
                "quality_class",
                "Not available"
            )

            confidence = classification.get("confidence")

            st.metric(
                "Predicted Water Quality",
                quality_class
            )

            if confidence is not None:

                st.metric(
                    "Classification Confidence",
                    f"{float(confidence) * 100:.2f}%"
                )

            probabilities = classification.get(
                "probabilities", {}
            )

            if probabilities:

                st.subheader(
                    "Classification Probabilities"
                )

                probability_df = pd.DataFrame(
                    list(probabilities.items()),
                    columns=[
                        "Water Quality Class",
                        "Probability"
                    ]
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

            # BEAS ANOMALY RESULT

            st.subheader("Beas River Anomaly Detection")

            anomaly = result.get(
                "anomaly_detection", {}
            )

            anomaly_status = anomaly.get(
                "status",
                "NOT_AVAILABLE"
            )

            if anomaly_status == "ANOMALY":

                st.error(
                    "Unusual water quality observation detected!"
                )

                st.metric(
                    "Anomaly Score",
                    anomaly.get("anomaly_score", "N/A")
                )

            elif anomaly_status == "NORMAL":

                st.success(
                    "No anomaly detected by the model."
                )

                st.metric(
                    "Anomaly Score",
                    anomaly.get("anomaly_score", "N/A")
                )

            elif anomaly_status == "NOT_AVAILABLE":

                st.warning(
                    "Anomaly detection is unavailable "
                    "for this reading."
                )

            else:

                st.warning(
                    f"Anomaly detection status: {anomaly_status}"
                )

            st.json(anomaly)

            st.subheader("Submitted Sensor Readings")

            st.json(
                result.get("input_data", payload)
            )

            st.caption(
                "An anomaly indicates an unusual observation "
                "according to the trained model. It does not "
                "independently establish pollution or unsafe water."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"API connection or prediction error: {error}"
            )

            st.info(
                "Make sure FastAPI is running at "
                "http://127.0.0.1:8002."
            )

        except (ValueError, TypeError) as error:

            st.error(
                f"Unable to interpret prediction response: {error}"
            )


# ============================================================
# 3. BEAS ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header("Beas River Anomaly Detection")

    st.write(
        "Historical anomaly detection results based on "
        "Beas River water quality observations."
    )

    df = load_csv(ANOMALY_FILE)

    if df is not None:

        st.success(
            "Anomaly detection results loaded!"
        )

        st.subheader("Dataset Overview")

        st.metric(
            "Total Records",
            len(df)
        )

        st.subheader("Anomaly Detection Data")

        st.dataframe(
            df,
            use_container_width=True
        )

        if "anomaly_status" in df.columns:

            status = (
                df["anomaly_status"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            normal_count = int(
                (status == "NORMAL").sum()
            )

            anomaly_count = int(
                (status == "ANOMALY").sum()
            )

            unknown_count = (
                len(df) - normal_count - anomaly_count
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Observations",
                len(df)
            )

            col2.metric(
                "Normal Readings",
                normal_count
            )

            col3.metric(
                "Anomalies Detected",
                anomaly_count
            )

            if len(df) > 0:

                percentage = (
                    anomaly_count / len(df)
                ) * 100

                st.metric(
                    "Anomaly Percentage",
                    f"{percentage:.2f}%"
                )

            if unknown_count > 0:

                st.warning(
                    f"{unknown_count} records have "
                    "unrecognized anomaly status."
                )

            st.subheader(
                "Anomaly Status Distribution"
            )

            st.bar_chart(status.value_counts())

            if "station_name" in df.columns:

                st.subheader(
                    "Station-wise Anomaly Distribution"
                )

                anomaly_df = df[
                    status == "ANOMALY"
                ]

                if not anomaly_df.empty:

                    station_counts = (
                        anomaly_df.groupby("station_name")
                        .size()
                        .sort_values(ascending=False)
                    )

                    st.bar_chart(station_counts)

                else:

                    st.info(
                        "No anomalies found in the dataset."
                    )

        else:

            st.warning(
                "The CSV does not contain an "
                "'anomaly_status' column."
            )

        if "date" in df.columns:

            st.subheader(
                "Historical Water Quality Trends"
            )

            trend_df = df.copy()

            trend_df["date"] = pd.to_datetime(
                trend_df["date"],
                errors="coerce"
            )

            trend_df = (
                trend_df
                .dropna(subset=["date"])
                .sort_values("date")
            )

            for parameter in [
                "pH",
                "Dissolved Oxygen"
            ]:

                if parameter in trend_df.columns:

                    chart_df = trend_df[
                        ["date", parameter]
                    ].copy()

                    chart_df[parameter] = pd.to_numeric(
                        chart_df[parameter],
                        errors="coerce"
                    )

                    chart_df = (
                        chart_df
                        .dropna()
                        .set_index("date")
                    )

                    if not chart_df.empty:

                        st.write(
                            f"{parameter} Trend"
                        )

                        st.line_chart(chart_df)

        st.download_button(
            "Download Anomaly Results",
            data=df.to_csv(index=False),
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            f"Anomaly results file not found: {ANOMALY_FILE}"
        )

        st.info(
            "Historical results are unavailable. "
            "You can still perform live anomaly detection "
            "through the Water Quality Prediction module."
        )


# ============================================================
# 4. FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header(
        "Beas River Water Quality Forecasting"
    )

    st.write(
        "Historical forecasting results and model evaluation."
    )

    forecast_df = load_csv(FORECAST_FILE)

    evaluation_df = load_csv(EVALUATION_FILE)

    if forecast_df is not None:

        st.success(
            "Forecasting results loaded successfully!"
        )

        st.subheader("Forecast Dataset")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        required_columns = {
            "indicator",
            "year",
            "forecast_value"
        }

        if required_columns.issubset(
            forecast_df.columns
        ):

            indicators = (
                forecast_df["indicator"]
                .dropna()
                .unique()
                .tolist()
            )

            if indicators:

                selected_indicator = st.selectbox(
                    "Select Water Quality Indicator",
                    indicators
                )

                selected_df = forecast_df[
                    forecast_df["indicator"]
                    == selected_indicator
                ].copy()

                selected_df["year"] = pd.to_numeric(
                    selected_df["year"],
                    errors="coerce"
                )

                selected_df["forecast_value"] = pd.to_numeric(
                    selected_df["forecast_value"],
                    errors="coerce"
                )

                selected_df = (
                    selected_df
                    .dropna(
                        subset=["year", "forecast_value"]
                    )
                    .sort_values("year")
                )

                st.subheader(
                    f"{selected_indicator} Forecast Trend"
                )

                if not selected_df.empty:

                    chart_df = selected_df.set_index(
                        "year"
                    )[["forecast_value"]]

                    st.line_chart(chart_df)

                    st.metric(
                        "Final Forecast Value",
                        f"{selected_df['forecast_value'].iloc[-1]:.3f}"
                    )

                else:

                    st.warning(
                        "No valid forecast values are available "
                        "for this indicator."
                    )

            else:

                st.warning(
                    "No forecast indicators were found."
                )

        else:

            st.warning(
                "Forecast CSV is missing one or more required "
                "columns: indicator, year, forecast_value."
            )

        st.download_button(
            "Download Forecast Results",
            data=forecast_df.to_csv(index=False),
            file_name="forecast_2024_2026.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            f"Forecast results file not found: {FORECAST_FILE}"
        )

        st.info(
            "Forecasting results will appear here once "
            "the forecasting pipeline generates the CSV file."
        )

    st.subheader("Forecast Model Evaluation")

    if evaluation_df is not None:

        st.dataframe(
            evaluation_df,
            use_container_width=True
        )

        st.caption(
            "MAE: Mean Absolute Error | "
            "RMSE: Root Mean Squared Error"
        )

        st.download_button(
            "Download Evaluation Results",
            data=evaluation_df.to_csv(index=False),
            file_name="forecast_evaluation.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    st.caption(
        "This section displays saved historical forecast results. "
        "It does not itself train forecasting models."
    )


# ============================================================
# 5. SENSOR HEALTH MONITORING
# ============================================================

elif page == "Sensor Health":

    st.header(
        "Beas River Sensor Health Monitoring"
    )

    st.write(
        "Historical monitoring data quality assessment."
    )

    df = load_csv(SENSOR_FILE)

    if df is not None:

        st.success(
            "Sensor health report loaded successfully!"
        )

        numeric_columns = [
            "total_records",
            "missing_readings",
            "invalid_readings",
            "repeated_values",
            "date_gaps_over_60_days"
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0)

        st.subheader("Sensor Health Overview")

        col1, col2, col3 = st.columns(3)

        if "station_name" in df.columns:

            col1.metric(
                "Monitoring Stations",
                df["station_name"].nunique()
            )

        if "missing_readings" in df.columns:

            col2.metric(
                "Missing Readings",
                int(df["missing_readings"].sum())
            )

        if "invalid_readings" in df.columns:

            col3.metric(
                "Invalid Readings",
                int(df["invalid_readings"].sum())
            )

        col4, col5 = st.columns(2)

        if "repeated_values" in df.columns:

            col4.metric(
                "Repeated Values",
                int(df["repeated_values"].sum())
            )

        if "date_gaps_over_60_days" in df.columns:

            col5.metric(
                "Date Gaps Over 60 Days",
                int(df["date_gaps_over_60_days"].sum())
            )

        st.subheader("Sensor Health Report")

        st.dataframe(
            df,
            use_container_width=True
        )

        quality_columns = [
            "missing_readings",
            "invalid_readings",
            "repeated_values"
        ]

        available_quality_columns = [
            column for column in quality_columns
            if column in df.columns
        ]

        if (
            "parameter" in df.columns
            and available_quality_columns
        ):

            st.subheader(
                "Parameter-wise Data Quality"
            )

            summary = df.groupby("parameter")[
                available_quality_columns
            ].sum()

            st.bar_chart(summary)

        if (
            "station_name" in df.columns
            and "missing_readings" in df.columns
        ):

            st.subheader(
                "Station-wise Missing Readings"
            )

            station_summary = (
                df.groupby("station_name")[
                    "missing_readings"
                ].sum()
            )

            st.bar_chart(station_summary)

        st.download_button(
            "Download Sensor Health Report",
            data=df.to_csv(index=False),
            file_name="sensor_health_report.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            f"Sensor health report not found: {SENSOR_FILE}"
        )


# ============================================================
# 6. SHAP EXPLAINABILITY
# ============================================================

elif page == "SHAP Explainability":

    st.header(
        "AquaSentinel SHAP Explainability"
    )

    st.write(
        "Understanding the contribution of input features "
        "to saved water quality model predictions."
    )

    shap_path = next(
        (
            path for path in SHAP_FILES
            if path.exists()
        ),
        None
    )

    if shap_path is not None:

        df = load_csv(shap_path)

        if df is not None:

            st.success(
                f"SHAP results loaded from {shap_path.name}"
            )

            required_columns = {
                "feature",
                "shap_value_class_1",
                "source_row_index"
            }

            if required_columns.issubset(
                df.columns
            ):

                df["shap_value_class_1"] = pd.to_numeric(
                    df["shap_value_class_1"],
                    errors="coerce"
                )

                df = df.dropna(
                    subset=[
                        "feature",
                        "shap_value_class_1"
                    ]
                )

                st.subheader(
                    "SHAP Explanation Data"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.subheader(
                    "Global Feature Importance"
                )

                importance = (
                    df.groupby("feature")[
                        "shap_value_class_1"
                    ]
                    .apply(
                        lambda values: values.abs().mean()
                    )
                    .sort_values(ascending=False)
                )

                st.bar_chart(importance)

                st.caption(
                    "Feature importance is represented by "
                    "mean absolute SHAP contribution across "
                    "the saved observations."
                )

                st.subheader(
                    "Local Prediction Explanation"
                )

                observation_ids = (
                    df["source_row_index"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if observation_ids:

                    selected_id = st.selectbox(
                        "Select Observation",
                        observation_ids
                    )

                    selected_df = df[
                        df["source_row_index"] == selected_id
                    ]

                    st.write(
                        f"Explanation for observation {selected_id}"
                    )

                    st.dataframe(
                        selected_df,
                        use_container_width=True
                    )

                    contributions = (
                        selected_df.groupby("feature")[
                            "shap_value_class_1"
                        ].sum()
                    )

                    st.subheader(
                        "Feature Contributions"
                    )

                    st.bar_chart(contributions)

                else:

                    st.warning(
                        "No valid observation IDs were found."
                    )

                st.info(
                    "Positive SHAP values contribute toward "
                    "the model's class 1 output; negative values "
                    "contribute in the opposite direction. "
                    "SHAP contributions explain model behavior, "
                    "not causality."
                )

                st.download_button(
                    "Download SHAP Results",
                    data=df.to_csv(index=False),
                    file_name="local_shap_contributions.csv",
                    mime="text/csv"
                )

            else:

                st.warning(
                    "The SHAP CSV does not contain all required "
                    "columns: feature, shap_value_class_1, "
                    "source_row_index."
                )

    else:

        st.warning(
            "Local SHAP results file not found."
        )

        st.info(
            "Generate the SHAP explanation results before "
            "using this dashboard section."
        )


# ============================================================
# 7. POLLUTION ALERTS
# ============================================================

elif page == "Pollution Alerts":

    st.header("Pollution Alerts")

    st.write(
        "Threshold-based water quality monitoring."
    )

    st.info(
        "This module uses configurable project demonstration "
        "thresholds. Set the thresholds according to the "
        "approved project methodology before interpreting "
        "real monitoring data."
    )

    st.subheader(
        "Enter Water Quality Parameters"
    )

    with st.form("pollution_alert_form"):

        col1, col2 = st.columns(2)

        with col1:

            ph = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=7.0
            )

            turbidity = st.number_input(
                "Turbidity (NTU)",
                min_value=0.0,
                value=4.0
            )

        with col2:

            dissolved_oxygen = st.number_input(
                "Dissolved Oxygen (mg/L)",
                min_value=0.0,
                value=8.0
            )

            conductivity = st.number_input(
                "Conductivity (uS/cm)",
                min_value=0.0,
                value=400.0
            )

        st.subheader(
            "Configurable Demonstration Thresholds"
        )

        ph_min = st.number_input(
            "Minimum pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        ph_max = st.number_input(
            "Maximum pH",
            min_value=0.0,
            max_value=14.0,
            value=8.5
        )

        do_min = st.number_input(
            "Minimum Dissolved Oxygen (mg/L)",
            min_value=0.0,
            value=5.0
        )

        turbidity_max = st.number_input(
            "Maximum Turbidity (NTU)",
            min_value=0.0,
            value=5.0
        )

        conductivity_max = st.number_input(
            "Maximum Conductivity (uS/cm)",
            min_value=0.0,
            value=1500.0
        )

        check_alerts = st.form_submit_button(
            "Check Pollution Alerts",
            use_container_width=True
        )

    if check_alerts:

        if ph_min > ph_max:

            st.error(
                "Minimum pH cannot be greater than maximum pH."
            )

        else:

            statuses = {}

            if ph < ph_min:

                statuses["pH"] = (
                    "Low",
                    f"pH is below the configured minimum ({ph_min})."
                )

            elif ph > ph_max:

                statuses["pH"] = (
                    "High",
                    f"pH is above the configured maximum ({ph_max})."
                )

            else:

                statuses["pH"] = (
                    "Normal",
                    "pH is within the configured range."
                )

            if dissolved_oxygen < do_min:

                statuses["Dissolved Oxygen"] = (
                    "Low",
                    "Dissolved Oxygen is below the configured minimum."
                )

            else:

                statuses["Dissolved Oxygen"] = (
                    "Normal",
                    "Dissolved Oxygen meets the configured minimum."
                )

            if turbidity > turbidity_max:

                statuses["Turbidity"] = (
                    "High",
                    "Turbidity exceeds the configured maximum."
                )

            else:

                statuses["Turbidity"] = (
                    "Normal",
                    "Turbidity is within the configured limit."
                )

            if conductivity > conductivity_max:

                statuses["Conductivity"] = (
                    "High",
                    "Conductivity exceeds the configured maximum."
                )

            else:

                statuses["Conductivity"] = (
                    "Normal",
                    "Conductivity is within the configured limit."
                )

            st.subheader(
                "Water Quality Alert Results"
            )

            abnormal_count = 0

            for parameter, (
                status,
                description
            ) in statuses.items():

                if status == "Normal":

                    st.success(
                        f"{parameter}: {status} — {description}"
                    )

                else:

                    abnormal_count += 1

                    st.warning(
                        f"{parameter}: {status} — {description}"
                    )

            if abnormal_count == 0:

                st.success(
                    "No configured threshold violations detected."
                )

            else:

                st.warning(
                    f"{abnormal_count} parameter(s) "
                    "exceeded configured thresholds."
                )

            st.caption(
                "These are deterministic threshold checks, "
                "not Isolation Forest model predictions. "
                "The demonstration thresholds are not "
                "independently verified regulatory limits."
            )


# ============================================================
# END OF AQUASENTINEL DASHBOARD
# ============================================================
