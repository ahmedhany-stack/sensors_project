🚀 Production-Grade MLOps Engine: Industrial Remaining Useful Life (RUL) Prediction System

An enterprise-ready, end-to-end MLOps platform built for predictive maintenance and Remaining Useful Life (RUL) estimation of high-value industrial assets (NASA C-MAPSS dataset).

This platform bridges the gap between offline Machine Learning models and production-grade software engineering. It incorporates real-time asynchronous inference, Triton Inference Server backend integration via gRPC, ONNX runtime optimization, distributed caching and feature store management via Redis, automated data drift detection, closed-loop model retraining (Continuous Training), real-time monitoring, and instant alert routing via Telegram.
🏛️ System Architecture & Data Flow
Plaintext

                        ┌─────────────────────────────────────────┐
                        │    Industrial Sensors & Devices         │
                        └────────────────────┬────────────────────┘
                                             │ (Batch/Stream REST Requests)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  FastAPI Microservice Engine                                            │
│                                                                                                         │
│  ┌──────────────────────┐     Cache Miss     ┌─────────────────────┐     gRPC Request     ┌───────────┐ │
│  │ Rate Limit & Auth    ├───────────────────►│ FastAPI Application ├─────────────────►│   Triton  │ │
│  └──────────┬───────────┘                    └──────────┬──────────┘                  │ Inference │ │
└─────────────┼───────────────────────────────────────────┼────────────────────────────►│  Server   │ │
              │                                           │ Async Bulk DB Logging       │  (ONNX)   │ │
              │                                           ▼                             └─────┬─────┘ │
              │                               ┌──────────────────────────────┐                │       │
              │                               │ PostgreSQL (`rul_db`)        │◄───────────────┘       │
              │                               │ - Bulk Prediction Persistence│                        │
              │ (Telemetry Metrics)           └──────────────┬───────────────┘                        │
              ▼                                              │                                        │
┌─────────────────────────┐                   ┌──────────────┴───────────────┐                        │
│ Prometheus & Grafana    │                   │ Apache Airflow Orchestration │                        │
└─────────────────────────┘                   └──────────────┬───────────────┘                        │
                                                             ▼                                        │
                                              ┌──────────────────────────────┐                        │
                                              │ Evidently AI Drift Engine    │                        │
                                              └──────────────┬───────────────┘                        │
                                                             │                                        │
                                              ┌──────────────┴───────────────┐                        │
                                              ▼ (Drift Identified)           ▼ (No Drift)             │
                               ┌──────────────────────────────┐     ┌─────────────────┐               │
                               │ Dynamic Retraining Pipeline  │     │ Pipeline Status │               │
                               │ & MLflow Artifact Registry   │     │ OK              │               │
                               └──────────────┬───────────────┘     └─────────────────┘               │
                                              │                                                       │
                                              ▼                                                       │
                               ┌──────────────────────────────┐     ┌─────────────────┐               │
                               │ Telegram Alert Notification  ├─────►│ Streamlit Live  │               │
                               │ (Instant Telemetry Dispatch) │     │ Observability   │               │
                               └──────────────────────────────┘     └─────────────────┘               │

🌟 Key Engineering & MLOps Highlights
⚡ 1. High-Performance Triton, ONNX & Redis Feature Store Integration

    Sub-Millisecond gRPC Inference: Replaced local model execution with Triton Inference Server backed by optimized ONNX Runtime (.onnx models). Communication via gRPC (HTTP/2) completely bypasses Python's GIL restrictions, maximizing inference throughput.

    Triton Dynamic Batcher: Configured automated request batching at the C++ server level (max_batch_size and max_queue_delay), optimizing hardware utilization under high industrial loads.

    Async Redis Feature Store (RedisFeatureStore): Integrated real-time sliding window and lag feature calculation directly backed by Redis, running fully asynchronously (await) inside FastAPI to prevent any event-loop blocking.

    Distributed Caching: Intercepts duplicated sensor signatures to eliminate redundant compute cycles and minimize response latencies.

💾 2. Asynchronous Bulk Persistence

    Non-Blocking Database Logging: Designed high-throughput asynchronous background queue workers (queue_worker.py) to batch and flush telemetry and inference records (handling heavy batches of 100 to 500+ items concurrently) to PostgreSQL.

🔄 3. Continuous Monitoring & Automated Training (Airflow + Evidently AI)

    Automated Data Drift Analysis: Scheduled Apache Airflow DAGs continuously aggregate production inference logs over sliding windows, running Kolmogorov-Smirnov statistical distribution tests using Evidently AI against baseline training datasets.

    Closed-Loop Retraining (CT): Automatically triggers MLflow tracking, artifact versioning, and checkpoint deployment upon detecting distribution drift (dataset_drift = True).

📊 4. Observability, Security & Alerting (Prometheus, Grafana, Telegram)

    Production Telemetry: Exposed Prometheus metrics tracking endpoint latency histograms, request throughput, error rates, and cache hits/misses, visualized on Grafana dashboards.

    Instant Incident Dispatch: Automated Telegram bot notifications for system drift alerts and execution summaries.

🛠️ Technology Stack
Domain	Tools & Frameworks
Core & Inference Engine	Python 3.10+, Triton Inference Server, ONNX Runtime, XGBoost, Pandas, NumPy
API & Service Layer	FastAPI, Uvicorn, Pydantic, OAuth2 Security, gRPC (tritonclient), Redis
Database & Caching	PostgreSQL, Redis, SQLAlchemy (ORM)
Orchestration & Workflow	Apache Airflow (TaskFlow API)
Experimentation & Tracking	MLflow, DVC (Data Version Control)
Monitoring & Drift	Evidently AI, Prometheus, Grafana
User Interface & Alerts	Streamlit, Telegram Bot API
Quality & CI/CD / Orchestration	Pytest, GitHub Actions, Docker, Docker Compose, Kubernetes (In Progress)
📂 Repository Structure
Plaintext

.
├── airflow/                    # Airflow DAGs & Automated Orchestration
│   └── dags/
│       └── rul_data_drift_dag.py
├── api/                        # FastAPI Application Modules
│   ├── app.py                  # Core Application & Endpoint Handlers
│   ├── database.py             # Database Connection & Session Management
│   └── queue_worker.py         # Asynchronous Background Queue Worker
├── models/                     # Triton Model Repository
│   └── engine_rul_model/
│       ├── 1/
│       │   └── full_rul_pipeline.onnx
│       └── config.pbtxt        # Triton Dynamic Batcher & Config
├── monitoring/                 # Drift Detection & Metrics Evaluators
│   └── monitoring.py           # Evidently AI Analysis Configurations
├── src/                        # ML Pipelines & Business Logic
│   ├── components/
│   │   └── data_transformation.py
│   └── utils/
│       └── feature_store.py    # Async Redis Feature Store Implementation
├── streamlit_app/              # Interactive Real-Time Monitoring Dashboard
│   └── app.py
├── tests/                      # Testing Suite
│   └── integration/
│       └── test_api.py         # End-to-End API Integration Tests
├── k8s/                        # Kubernetes Deployments, Services & ConfigMaps (WIP)
├── docker-compose.yml          # Multi-Container Deployment Orchestration
├── Dockerfile                  # Application Container Specification
└── requirements.txt            # Python Dependencies

⚡ Quickstart Guide
Prerequisites

    Docker and Docker Compose

    Git

1. Clone Repository & Setup Environment
Bash

git clone https://github.com/ahmedhany-stack/sensors_project.git
cd sensors_project

Create a .env configuration file in the project root:
مقتطف الرمز

# Database Credentials
DB_USER=airflow
DB_PASSWORD=airflow
DB_HOST=postgres
DB_PORT=5432
DB_NAME=rul_db

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_HOST=redis
REDIS_PORT=6379

# Internal Security
INTERNAL_API_SECRET=super-secret-key

# Alerting Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

2. Launch Entire Infrastructure with Docker Compose

Spin up Triton Inference Server, FastAPI, PostgreSQL, Redis, Apache Airflow, Prometheus, Grafana, and Streamlit with a single command:
Bash

docker compose up --build -d

3. Verify System Services
Service Name	Web Interface / Endpoint	Default URL
FastAPI Documentation	Swagger UI	http://localhost:8000/docs
Triton Inference Server	gRPC / HTTP	http://localhost:8001
Streamlit Dashboard	Live Operations & Curves	http://localhost:8501
Apache Airflow	DAG Orchestration Platform	http://localhost:8080
Prometheus	Metric Aggregation Engine	http://localhost:9090
Grafana	Live Monitoring Dashboard	http://localhost:3000
☸️ What's Next? (Kubernetes & Cloud Native Migration)

We are currently scaling this architecture to a Production Kubernetes Cluster setup:

    Deployments & StatefulSets: Managing stateless FastAPI/Streamlit microservices alongside stateful PostgreSQL and Redis storage pods.

    Services & Ingress: Exposing internal gRPC ports for Triton and handling external ingress traffic routing.

    ConfigMaps & Secrets: Secure environment variable and credential management across cloud pods.

👤 Author & Maintainer: Ahmed Hany Sallam (أحمد هاني سلام)

    GitHub: @ahmedhany-stack

    Role: Machine Learning & MLOps Engineer

    Developed with a focus on enterprise architecture, low-latency gRPC inference, high-throughput asynchronous feature stores, and automated predictive maintenance.