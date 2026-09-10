import os
import time
import json
import hashlib
import asyncio
import logging
import yaml
from contextlib import asynccontextmanager
from typing import Dict, Any

import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, status, Response, BackgroundTasks, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Prometheus Imports
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.api.schemas import (
    BatchPredictionInput, 
    PredictionResponse, 
    SinglePredictionOutput, 
    HealthCheckResponse,
    Token,
    TokenData
)
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.api.database import save_predictions_to_db

# AUTHENTICATION & AUTHORIZATION IMPORTS
from src.api.auth import create_access_token, verify_password, hash_password
from src.api.dependencies import get_current_user, require_role


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل كافة الإعدادات
config = load_config()
app_cfg = config.get("app", {})
redis_cfg = config.get("redis", {})
reports_cfg = config.get("reports", {})
users_cfg = config.get("users", {}).get("fake_db", {})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_API")

# ==========================================
# MOCK USER DATABASE (توليد المستخدمين مع تصحيح جلب الباسورد)
# ==========================================
FAKE_USERS_DB = {}
for u_key, u_data in users_cfg.items():
    env_var_name = u_data.get("password_env_var")
    
    # جلب الباسورد من الـ ENV إذا وُجدت وقيمتها ليست فارغة، وإلا استخدام الـ default_password
    env_val = os.getenv(env_var_name) if env_var_name else None
    raw_pwd = env_val if env_val else u_data.get("default_password")

    username = u_data.get("username", u_key)

    FAKE_USERS_DB[username] = {
        "username": username,
        "hashed_password": hash_password(str(raw_pwd)),
        "role": u_data.get("role")
    }

# ضمان وجود حساب admin احتياطي دائماً لبيئة التطوير
if "admin" not in FAKE_USERS_DB:
    FAKE_USERS_DB["admin"] = {
        "username": "admin",
        "hashed_password": hash_password("admin123"),
        "role": "admin"
    }

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

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total Cache Hits in Redis",
    ["status"]
)

ml_artifacts: Dict[str, Any] = {}

# ==========================================
# REDIS CONFIGURATION
# ==========================================
REDIS_HOST = os.getenv("REDIS_HOST", redis_cfg.get("host", "redis"))
REDIS_PORT = int(os.getenv("REDIS_PORT", redis_cfg.get("port", 6379)))
CACHE_EXPIRATION_SECONDS = redis_cfg.get("cache_expiration_seconds", 3600)
CACHE_PREFIX = redis_cfg.get("cache_prefix", "rul_cache")


def generate_cache_key(records_list: list) -> str:
    serialized_data = json.dumps(records_list, sort_keys=True)
    hash_value = hashlib.md5(serialized_data.encode('utf-8')).hexdigest()
    return f"{CACHE_PREFIX}:{hash_value}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. تحميل الـ ML Pipeline
    try:
        ml_artifacts["pipeline"] = PredictionPipeline()
        ml_artifacts["model_loaded"] = True
        logger.info("Prediction Pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize artifacts: {str(e)}")
        ml_artifacts["model_loaded"] = False

    # 2. إنشاء الاتصال بـ Redis Server
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=0, 
            decode_responses=True,
            socket_connect_timeout=2
        )
        redis_client.ping()
        ml_artifacts["redis"] = redis_client
        logger.info(f"Connected to Redis successfully at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis cache: {str(e)}. Proceeding without caching.")
        ml_artifacts["redis"] = None

    yield

    if ml_artifacts.get("redis"):
        ml_artifacts["redis"].close()
    ml_artifacts.clear()
    logger.info("API server shutdown complete.")


app = FastAPI(
    title=app_cfg.get("title", "Engine RUL API"), 
    version=app_cfg.get("version", "1.0.0"),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to Engine RUL Prediction API",
        "swagger_docs": "/docs",
        "metrics": "/metrics",
        "cache_enabled": ml_artifacts.get("redis") is not None
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    is_loaded = ml_artifacts.get("model_loaded", False)
    return HealthCheckResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        version=app_cfg.get("version", "1.0.0")
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(
    payload: BatchPredictionInput, 
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    start_time = time.time()

    if not ml_artifacts.get("model_loaded", False):
        REQUEST_COUNT.labels(endpoint="/predict", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        input_data = [record.dict() for record in payload.records]
        redis_client: redis.Redis = ml_artifacts.get("redis")
        cache_key = generate_cache_key(input_data)
        
        cached_predictions = None

        if redis_client:
            try:
                cached_data = redis_client.get(cache_key)
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
                    redis_client.setex(
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

        output_predictions = [
            SinglePredictionOutput(**pred) for pred in predictions
        ]

        REQUEST_COUNT.labels(endpoint="/predict", status_code="200").inc()
        return PredictionResponse(
            status="success", 
            total_records=len(output_predictions), 
            predictions=output_predictions
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status_code="500").inc()
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)


@app.get("/monitoring/report", tags=["Monitoring"])
async def get_drift_report(
    current_user: TokenData = Depends(require_role(["admin"]))
):
    report_path = reports_cfg.get("drift_report_path", os.path.join("reports", "drift_report.html"))
    if os.path.exists(report_path):
        return FileResponse(report_path)
    raise HTTPException(status_code=404, detail="Report not generated yet.")