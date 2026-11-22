
"""
AquaSentinel - FastAPI Prediction Server

Endpoints:
    POST /predict
    GET /forecast
    GET /forecast/all
    GET /forecast/parameters
    GET /health
    GET /docs

Models:
    1. Water Quality Classification
    2. Beas River Anomaly Detection (pH + Dissolved Oxygen)
    3. Time-Series Forecasting
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.classifier import classifier
from src.beas_anomaly_inference import beas_anomaly_detector


# APPLICATION SETUP

app = FastAPI(
    title="AquaSentinel Water Quality ML API",
    description=(
        "Water quality prediction API for the AquaSentinel monitoring system."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST MODEL

class SensorReading(BaseModel):
    ph: float = Field(..., ge=0, le=14)

    Hardness: float = Field(default=200, ge=0)
    Solids: float = Field(default=20000, ge=0)
    Chloramines: float = Field(default=7, ge=0)
    Sulfate: float = Field(default=300, ge=0)
    Conductivity: float = Field(default=400, ge=0)
    Organic_carbon: float = Field(default=14, ge=0)
    Trihalomethanes: float = Field(default=60, ge=0)
    Turbidity: float = Field(default=4, ge=0)

    temperature: Optional[float] = None

    do: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved Oxygen in mg/L"
    )

    tds: Optional[float] = Field(default=None, ge=0)


# RESPONSE MODEL

class PredictionResponse(BaseModel):
    classification: dict
    anomaly_detection: dict
    input_data: dict


# MODEL STARTUP

@app.on_event("startup")
async def load_models():

    print("\nLoading AquaSentinel ML Models...")

    try:
        classifier.load()
        print("Classification model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Classifier not loaded: {e}")

    try:
        beas_anomaly_detector.load()
        print("Beas anomaly detector loaded successfully.")
    except Exception as e:
        print(f"WARNING: Beas anomaly detector not loaded: {e}")

    print("AquaSentinel API startup completed.\n")


# HEALTH CHECK

@app.get("/health", tags=["System"])
async def health_check():

    return {
        "status": "healthy",
        "models": {
            "classifier": classifier._loaded,
            "anomaly_detector": beas_anomaly_detector.loaded
        },
        "message": "AquaSentinel Water Quality API is running."
    }


# WATER QUALITY PREDICTION

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"]
)
async def predict(reading: SensorReading):

    features = reading.model_dump(exclude_none=True)

    # Classification
    try:
        classification_result = classifier.predict(features)
    except Exception as e:
        classification_result = {
            "error": str(e)
        }

    # Beas anomaly detection
    if features.get("do") is None:

        anomaly_result = {
            "model": "Beas River Isolation Forest",
            "status": "NOT_AVAILABLE",
            "is_anomaly": None,
            "error": "Dissolved Oxygen (do) is required."
        }

    else:

        try:
            anomaly_result = beas_anomaly_detector.detect(features)
        except Exception as e:
            anomaly_result = {
                "model": "Beas River Isolation Forest",
                "status": "ERROR",
                "is_anomaly": None,
                "error": str(e)
            }

    return PredictionResponse(
        classification=classification_result,
        anomaly_detection=anomaly_result,
        input_data=features
    )


# SINGLE-PARAMETER FORECAST

@app.get("/forecast", tags=["Forecasting"])
async def forecast(
    parameter: str = Query(default="ph"),
    hours: int = Query(default=24, ge=1, le=168)
):

    try:
        from src.forecaster import forecaster
        return forecaster.forecast(parameter, hours)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FORECAST ALL PARAMETERS

@app.get("/forecast/all", tags=["Forecasting"])
async def forecast_all(
    hours: int = Query(default=24, ge=1, le=168)
):

    try:
        from src.forecaster import forecaster
        return forecaster.forecast_all(hours)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AVAILABLE FORECAST PARAMETERS

@app.get("/forecast/parameters", tags=["Forecasting"])
async def forecast_parameters():

    try:
        from src.forecaster import forecaster

        return {
            "available_parameters": (
                forecaster.get_available_parameters()
            )
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# RUN SERVER DIRECTLY

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )
