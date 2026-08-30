import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Prometheus Imports
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import BatchPredictionInput, PredictionResponse, SinglePredictionOutput, HealthCheckResponse
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.api.database import save_predictions_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_API")

# ==========================================
# PROMETHEUS METRICS DEFINITIONS
# ==========================================
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests",
    ["endpoint", "status_code"]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent processing inference request"
)

PREDICTED_RUL_GAUGE = Gauge(
    "model_last_predicted_rul",
    "Last predicted RUL value",
    ["unit_number"]
)

ml_artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_artifacts["pipeline"] = PredictionPipeline()
        ml_artifacts["model_loaded"] = True
        logger.info("Prediction Pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize artifacts: {str(e)}")
        ml_artifacts["model_loaded"] = False

    yield

    ml_artifacts.clear()
    logger.info("API server shutdown complete.")


app = FastAPI(title="Engine RUL API with Monitoring", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to Engine RUL Prediction API",
        "swagger_docs": "/docs",
        "metrics": "/metrics",
    }


# Endpoint خاص بـ Prometheus لمسح البيانات
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    is_loaded = ml_artifacts.get("model_loaded", False)
    return HealthCheckResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(payload: BatchPredictionInput, background_tasks: BackgroundTasks):
    start_time = time.time()

    if not ml_artifacts.get("model_loaded", False):
        REQUEST_COUNT.labels(endpoint="/predict", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        # حيلة للتست: لو أرسلنا unit_number = 999 خليه ينام 1.5 ثانية
        if any(record.unit_number == 999 for record in payload.records):
            await asyncio.sleep(1.5)

        input_data = [record.dict() for record in payload.records]
        df = pd.DataFrame(input_data)

        # التوقع
        pipeline: PredictionPipeline = ml_artifacts["pipeline"]
        raw_predictions = pipeline.predict(df)

        # حفظ البيانات في PostgreSQL في الخلفية لعدم إبطاء الـ Response
        background_tasks.add_task(save_predictions_to_db, input_data, raw_predictions.tolist() if hasattr(raw_predictions, 'tolist') else list(raw_predictions))

        predictions = []
        for unit, cycle, pred in zip(df["unit_number"], df["time_in_cycles"], raw_predictions):
            PREDICTED_RUL_GAUGE.labels(unit_number=str(unit)).set(pred)
            predictions.append(SinglePredictionOutput(
                unit_number=int(unit),
                time_in_cycles=int(cycle),
                predicted_rul=round(float(pred), 2)
            ))

        REQUEST_COUNT.labels(endpoint="/predict", status_code="200").inc()
        return PredictionResponse(status="success", total_records=len(predictions), predictions=predictions)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status_code="500").inc()
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)


@app.get("/monitoring/report", tags=["Monitoring"])
async def get_drift_report():
    report_path = os.path.join("reports", "drift_report.html")
    if os.path.exists(report_path):
        return FileResponse(report_path)
    raise HTTPException(status_code=404, detail="Report not generated yet.")