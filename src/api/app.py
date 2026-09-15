import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import time
import logging
import hashlib
import json
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
import numpy as np
import orjson 
import redis.asyncio as aioredis

# ⚡ استيراد Triton gRPC Client الرسمي
import tritonclient.grpc.aio as grpcclient

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks, Header
from fastapi.responses import ORJSONResponse 
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.components.data_transformation import DataTransformation
from src.api.database import init_db, engine, Base
# استيراد الـ Worker الجديد
from src.api.queue_worker import prediction_worker

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_app")

# ------------------------------------------------------------------------------
# Settings & Configurations
# ------------------------------------------------------------------------------
REDIS_URL  = os.getenv("REDIS_URL", "redis://redis_cache:6379/0")
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001") 
MODEL_NAME = "engine_rul_model"
CACHE_EXPIRATION_SECONDS = 3600
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "super-secret-key")

# ------------------------------------------------------------------------------
# Prometheus Metrics
# ------------------------------------------------------------------------------
REQUEST_COUNT = Counter("app_requests_total", "Total app requests", ["endpoint", "status_code"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time taken for prediction")
CACHE_HITS = Counter("cache_hits_total", "Cache hits count", ["status"])
RATE_LIMIT_EXCEEDED = Counter("rate_limit_exceeded_total", "Total rate limit violations")
PREDICTED_RUL_GAUGE = Gauge("predicted_rul_value", "Predicted RUL value", ["unit_number"])

# ------------------------------------------------------------------------------
# Triton State & Artifacts
# ------------------------------------------------------------------------------
ml_artifacts = {}
from typing import Optional
# ...
redis_pool: Optional[aioredis.ConnectionPool] = None
data_transformer = DataTransformation()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool
    logger.info("Initializing Triton Client, Database, Redis and Queue Worker...")
    
    # 1. تحميل DataTransformation وقائمة الـ Features إلى الـ RAM مرة واحدة فقط
    try:
        ml_artifacts["data_transformer"] = data_transformer
        ml_artifacts["constant_cols"] = getattr(data_transformer, "constant_cols", ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19'])
        
        features_json_path = data_transformer.transformation_config.features_path
        if os.path.exists(features_json_path):
            with open(features_json_path, "r", encoding="utf-8") as f:
                ml_artifacts["feature_names"] = json.load(f)
            logger.info("Loaded feature names schema and transformer into RAM.")
        else:
            ml_artifacts["feature_names"] = None
    except Exception as feat_err:
        logger.warning(f"Failed to pre-load features list: {feat_err}")
        ml_artifacts["feature_names"] = None

    # 2. تهيئة الاتصال بـ Triton Inference Server عبر gRPC
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, ssl=False)
        if await triton_client.is_server_live():
            ml_artifacts["triton_client"] = triton_client
            ml_artifacts["model_loaded"] = True
            logger.info(f"Successfully connected to Triton Server at {TRITON_URL}")
        else:
            ml_artifacts["model_loaded"] = False
            logger.error("Triton Server is not live!")
    except Exception as triton_err:
        ml_artifacts["model_loaded"] = False
        logger.error(f"Failed to connect to Triton Inference Server: {triton_err}")

    # 3. إنتاج الجداول للـ DB
    try:
        init_db()
    except Exception as db_err:
        logger.error(f"Failed to initialize Database: {db_err}")

    # 4. تشغيل الـ Queue Worker الخلفي
    try:
        await prediction_worker.start()
    except Exception as worker_err:
        logger.error(f"Failed to start Prediction Queue Worker: {worker_err}")

    # 5. الاتصال بـ Redis باستخدام ConnectionPool ثابت
    try:
        redis_pool = aioredis.ConnectionPool.from_url(
            REDIS_URL, 
            max_connections=20, 
            decode_responses=False
        )
        redis = aioredis.Redis(connection_pool=redis_pool)
        await redis.ping()
        ml_artifacts["redis"] = redis
        logger.info("Connected to Redis successfully via ConnectionPool.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        ml_artifacts["redis"] = None

    # 6. Warm-up Inference لـ Triton
    if ml_artifacts.get("model_loaded", False):
        try:
            logger.info("Executing Triton Warm-up Inference...")
            dummy_data = np.zeros((1, 68), dtype=np.float32)
            inputs = [grpcclient.InferInput("float_input", dummy_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(dummy_data)
            outputs = [grpcclient.InferRequestedOutput("xgb_variable")]
            
            client: grpcclient.InferenceServerClient = ml_artifacts["triton_client"]
            _ = await client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            logger.info("Triton warm-up inference completed successfully.")
        except Exception as warm_err:
            logger.warning(f"Triton warm-up inference failed: {warm_err}")

    yield

    # Shutdown
    logger.info("Shutting down resources...")
    try:
        await prediction_worker.stop()
    except Exception as stop_err:
        logger.error(f"Error stopping queue worker: {stop_err}")

    client = ml_artifacts.get("triton_client")
    if client:
        await client.close()
    
    redis = ml_artifacts.get("redis")
    if redis:
        await redis.close()
    if redis_pool:
        await redis_pool.disconnect()
        logger.info("Redis ConnectionPool disconnected.")

# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API with Triton",
    version="2.0.0",
    default_response_class=ORJSONResponse,
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
    serialized = orjson.dumps(input_data)
    return f"cache:predict:{hashlib.md5(serialized).hexdigest()}"

from src.utils.feature_store import RedisFeatureStore

redis_host = os.getenv("REDIS_HOST", "redis") # أو redis_cache حسب الشبكة
redis_port = int(os.getenv("REDIS_PORT", 6379))
feature_store = RedisFeatureStore(host=redis_host, port=redis_port, window_size=10)
async def prepare_features_with_redis(input_data: list, feature_names: list, constant_cols: list):
    df_raw = pd.DataFrame(input_data)
    
    safe_constant_cols = constant_cols if constant_cols is not None else []
    cols_to_drop = [c for c in safe_constant_cols if c in df_raw.columns]
    if cols_to_drop:
        df_raw.drop(columns=cols_to_drop, inplace=True)
        
    sensor_cols = [c for c in df_raw.columns if c.startswith('s_') or c.startswith('setting_')]

    # 🚀 إضافة await هنا لتفادي الـ Blocking تماماً
    processed_rows = await feature_store.get_rolling_and_lags_batch(input_data, sensor_cols)

    df_features = pd.DataFrame(processed_rows)
    
    if feature_names:
        df_model_input = df_features.reindex(columns=feature_names, fill_value=0.0)
    else:
        df_model_input = df_features

    X_raw = df_model_input.fillna(0.0).astype("float32").values
    
    if X_raw.shape[1] > 68:
        X_raw = X_raw[:, :68]
    elif X_raw.shape[1] < 68:
        pad_width = 68 - X_raw.shape[1]
        X_raw = np.pad(X_raw, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
    
    return df_raw, X_raw



#----------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.post("/token", tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "fake-token", "token_type": "bearer"}

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

import anyio

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(
    request: Request,
    payload: BatchPredictionInput, 
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    start_time = time.time()
    
    if not ml_artifacts.get("model_loaded", False) or ml_artifacts.get("triton_client") is None:
        REQUEST_COUNT.labels(endpoint="/predict", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Triton Server client unavailable")

    try:
        input_data = [record.model_dump() for record in payload.records]
        if not input_data:
            raise HTTPException(status_code=400, detail="Input records list cannot be empty.")

        redis_client: aioredis.Redis = ml_artifacts.get("redis")
        cache_key = generate_cache_key(input_data)
        cached_predictions = None

        if redis_client:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    cached_predictions = orjson.loads(cached_data)
                    CACHE_HITS.labels(status="hit").inc()
            except Exception as cache_err:
                logger.warning(f"Error reading from Redis cache: {cache_err}")

        if cached_predictions is None:
            if redis_client:
                CACHE_HITS.labels(status="miss").inc()

            feature_names = ml_artifacts.get("feature_names")
            constant_cols_val = ml_artifacts.get("constant_cols") or []
            
# استدعاء دالة تحضير الميزات المتزامنة (Async) مباشرة بدون anyio
            df_raw, np_data = await prepare_features_with_redis(
                input_data,
                feature_names,
                constant_cols_val
            )

            triton_client: grpcclient.InferenceServerClient = ml_artifacts["triton_client"]
            
            inputs = [grpcclient.InferInput("float_input", np_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(np_data)
            outputs = [grpcclient.InferRequestedOutput("xgb_variable")]

            response = await triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            raw_preds_array = response.as_numpy("xgb_variable")
            raw_preds_list = raw_preds_array.flatten().tolist()

            for rec, pred in zip(input_data, raw_preds_list):
                await prediction_worker.put(
                    input_data=rec,
                    prediction=float(pred),
                    unit_number=rec.get("unit_number"),
                    time_in_cycles=rec.get("time_in_cycles")
                )

            predictions = []
            for unit, cycle, pred in zip(df_raw["unit_number"], df_raw["time_in_cycles"], raw_preds_list):
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
                        orjson.dumps(predictions)
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
        logger.error(f"Error during Triton inference: {str(e)}", exc_info=True)
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
        "message": "Predictive Maintenance API with Triton Server is running smoothly.",
        "swagger_docs": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    is_triton_live = False
    try:
        client = ml_artifacts.get("triton_client")
        if client:
            is_triton_live = await client.is_server_live()
    except Exception:
        is_triton_live = False
    
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

    overall_status = "healthy" if (is_triton_live and db_status == "healthy") else "degraded"

    return {
        "status": overall_status,
        "version": "2.0.0",
        "triton_server_live": is_triton_live,
        "services": {
            "database": db_status,
            "redis": redis_status
        }
    }