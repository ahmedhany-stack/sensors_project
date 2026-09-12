import os
import time
import json
import logging
import hashlib
import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ✅ استيراد عناصر قاعدة البيانات من database.py لمنع التكرار ولضمان عزل الاتصال
from src.api.database import init_db, engine, Base, save_predictions_to_db

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_app")

# ------------------------------------------------------------------------------
# Settings & Configurations
# ------------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_EXPIRATION_SECONDS = 3600

# ------------------------------------------------------------------------------
# Prometheus Metrics
# ------------------------------------------------------------------------------
REQUEST_COUNT = Counter("app_requests_total", "Total app requests", ["endpoint", "status_code"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time taken for prediction")
CACHE_HITS = Counter("cache_hits_total", "Cache hits count", ["status"])
RATE_LIMIT_EXCEEDED = Counter("rate_limit_exceeded_total", "Total rate limit violations")
PREDICTED_RUL_GAUGE = Gauge("predicted_rul_value", "Predicted RUL value", ["unit_number"])

# ------------------------------------------------------------------------------
# Dummy Pipeline & Artifacts State
# ------------------------------------------------------------------------------
class PredictionPipeline:
    def predict(self, df: pd.DataFrame):
        return [100.5 - i for i in range(len(df))]

ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: إعداد الموديل والـ Redis والداتابيز
    logger.info("Initializing ML Artifacts, Database and Redis Connection...")
    ml_artifacts["pipeline"] = PredictionPipeline()
    ml_artifacts["model_loaded"] = True
    
    # ✅ استدعاء إنشاء الجداول عند قومة السيرفر فقط وليس عند الـ Import
    try:
        init_db()
    except Exception as db_err:
        logger.error(f"Failed to initialize Database: {db_err}")

    try:
        redis = aioredis.from_url(REDIS_URL, decode_responses=False)
        await redis.ping()
        ml_artifacts["redis"] = redis
        logger.info("Connected to Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        ml_artifacts["redis"] = None

    yield

    # Shutdown: إغلاق الـ Redis
    logger.info("Shutting down resources...")
    redis = ml_artifacts.get("redis")
    if redis:
        await redis.close()

# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class SingleRecordInput(BaseModel):
    unit_number: int
    time_in_cycles: int
    setting_1: float = 0.0
    setting_2: float = 0.0
    setting_3: float = 0.0
    s_1: float = 0.0
    s_2: float = 0.0
    s_3: float = 0.0
    s_4: float = 0.0
    s_5: float = 0.0
    s_6: float = 0.0
    s_7: float = 0.0
    s_8: float = 0.0
    s_9: float = 0.0
    s_10: float = 0.0
    s_11: float = 0.0
    s_12: float = 0.0
    s_13: float = 0.0
    s_14: float = 0.0
    s_15: float = 0.0
    s_16: float = 0.0
    s_17: float = 0.0
    s_18: float = 0.0
    s_19: float = 0.0
    s_20: float = 0.0
    s_21: float = 0.0

class BatchPredictionInput(BaseModel):
    records: List[SingleRecordInput]

class SinglePredictionOutput(BaseModel):
    unit_number: int
    time_in_cycles: int
    predicted_rul: float

class PredictionResponse(BaseModel):
    status: str
    total_records: int
    predictions: List[SinglePredictionOutput]

class TokenData(BaseModel):
    username: str

# ------------------------------------------------------------------------------
# Authentication Dependencies
# ------------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return TokenData(username="admin")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def generate_cache_key(input_data: list) -> str:
    serialized = json.dumps(input_data, sort_keys=True)
    return f"cache:predict:{hashlib.md5(serialized.encode()).hexdigest()}"

# ------------------------------------------------------------------------------
# Rate Limit & Maintenance Verification
# ------------------------------------------------------------------------------
async def check_rate_limit_and_maintenance(client_ip: str, limit: int = 60, window: int = 60):
    redis_client: aioredis.Redis = ml_artifacts.get("redis")
    if not redis_client:
        return

    is_maintenance = await redis_client.get("app:maintenance")
    if is_maintenance is not None:
        val = is_maintenance.decode('utf-8') if isinstance(is_maintenance, bytes) else str(is_maintenance)
        if val.lower() == "true":
            REQUEST_COUNT.labels(endpoint="/predict", status_code="503").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System is currently under maintenance. Please try again later."
            )

    rate_key = f"rate_limit:{client_ip}"
    current_requests = await redis_client.incr(rate_key)
    if current_requests == 1:
        await redis_client.expire(rate_key, window)
    if current_requests > limit:
        RATE_LIMIT_EXCEEDED.inc()
        REQUEST_COUNT.labels(endpoint="/predict", status_code="429").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Too many requests in a short time."
        )

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.post("/token", tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "fake-token", "token_type": "bearer"}

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(
    request: Request,
    payload: BatchPredictionInput, 
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "127.0.0.1"

    await check_rate_limit_and_maintenance(client_ip)

    if not ml_artifacts.get("model_loaded", False):
        REQUEST_COUNT.labels(endpoint="/predict", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        input_data = [record.dict() for record in payload.records]
        if not input_data:
            raise HTTPException(status_code=400, detail="Input records list cannot be empty.")

        redis_client: aioredis.Redis = ml_artifacts.get("redis")
        cache_key = generate_cache_key(input_data)
        cached_predictions = None

        if redis_client:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    cached_predictions = json.loads(cached_data)
                    CACHE_HITS.labels(status="hit").inc()
                    logger.info("Cache Hit: Returning predictions directly from Redis.")
            except Exception as cache_err:
                logger.warning(f"Error reading from Redis cache: {cache_err}")

        if cached_predictions is None:
            if redis_client:
                CACHE_HITS.labels(status="miss").inc()

            df = pd.DataFrame(input_data)
            pipeline: PredictionPipeline = ml_artifacts["pipeline"]
            raw_predictions = pipeline.predict(df)
            
            raw_preds_list = raw_predictions.tolist() if hasattr(raw_predictions, 'tolist') else list(raw_predictions)

            background_tasks.add_task(save_predictions_to_db, input_data, raw_preds_list)

            predictions = []
            for unit, cycle, pred in zip(df["unit_number"], df["time_in_cycles"], raw_preds_list):
                predictions.append({
                    "unit_number": int(unit),
                    "time_in_cycles": int(cycle),
                    "predicted_rul": round(float(pred), 2)
                })

            if redis_client:
                try:
                    await redis_client.setex(
                        cache_key, 
                        CACHE_EXPIRATION_SECONDS, 
                        json.dumps(predictions)
                    )
                except Exception as cache_err:
                    logger.warning(f"Failed to set value in Redis: {cache_err}")
        else:
            predictions = cached_predictions

        for item in predictions:
            PREDICTED_RUL_GAUGE.labels(unit_number=str(item["unit_number"])).set(item["predicted_rul"])

        output_predictions = [SinglePredictionOutput(**pred) for pred in predictions]
        REQUEST_COUNT.labels(endpoint="/predict", status_code="200").inc()

        return PredictionResponse(
            status="success", 
            total_records=len(output_predictions), 
            predictions=output_predictions
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status_code="500").inc()
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

# ------------------------------------------------------------------------------
# Health & Root Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root(current_user: TokenData = Depends(get_current_user)):
    return {
        "message": "Predictive Maintenance API is running smoothly.",
        "swagger_docs": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    is_model_loaded = ml_artifacts.get("model_loaded", False)
    
    redis_status = "healthy"
    redis_client = ml_artifacts.get("redis")
    if redis_client:
        try:
            await redis_client.ping()
        except Exception:
            redis_status = "unhealthy"
    else:
        redis_status = "unreachable"

    db_status = "healthy"
    try:
        with engine.connect() as connection:
            pass
    except Exception:
        db_status = "unhealthy"

    overall_status = "healthy" if (is_model_loaded and db_status == "healthy") else "degraded"

    return {
        "status": overall_status,
        "version": "1.0.0",
        "model_loaded": is_model_loaded,
        "services": {
            "database": db_status,
            "redis": redis_status
        }
    }