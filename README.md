🚀 Enterprise-Grade RUL Prediction & MLOps Automated Pipeline

    A production-ready, end-to-end MLOps pipeline designed for Remaining Useful Life (RUL) prediction of industrial engines (CMAPSS dataset). Built to handle real-time inference, automated data drift detection, dynamic model retraining, and live alerting.

🏗️ System Architecture & Workflow
Plaintext

[ Industrial Engines / Sensors ] 
             │
             ▼ (REST API - POST)
     [ FastAPI Service ] ──> Logs Predictions ──> [ PostgreSQL (`rul_db`) ]
             │                                          │
             │ (Async Log Capture)                      │ (Data Window: Last 24 Hours)
             ▼                                          ▼
   [ Streamlit Dashboard ] <───────────────── [ Apache Airflow Orchestration ]
     (Live RUL, Alerts & Metrics)                       │
                                                        ▼
                                           [ Evidently AI Drift Analysis ]
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼ (Drift Detected: True)    ▼ (No Drift)
                               [ Automated Retraining ]       [ Pipeline Success ]
                                          │
                                          ▼
                               [ Telegram Alert Notification ]

🛠️ Tech Stack & Ecosystem

    Core & Pipeline: Python, Pandas, Scikit-XGBoost / ML Models.

    API & Serving: FastAPI, Uvicorn, Pydantic.

    Database & Storage: PostgreSQL (rul_db), DVC (Data Version Control).

    Orchestration & Automation: Apache Airflow (DAGs, TaskFlow API).

    Monitoring & Observability: Evidently AI (Data Drift Detection), Prometheus, Grafana.

    Frontend / Dashboard: Streamlit (Real-time monitoring, alerts, and RUL degradation curves).

    Alerting System: Telegram Bot API (Automated critical failure and retraining alerts).

    Testing & Quality Assurance: Pytest (Unit tests for data processing/models + Integration tests for API).

    Containerization: Docker, Docker Compose, WSL2.

📂 Project Directory Structure
Plaintext

.
├── airflow/                    # Airflow DAGs and orchestration scripts
│   └── dags/
│       └── rul_data_drift_dag.py
├── api/                        # FastAPI application and routing
│   ├── main.py
│   └── schemas.py
├── data/                       # Datasets (Raw, Processed, and DVC tracked)
│   ├── raw/
│   └── processed/
├── monitoring/                 # Evidently AI configuration and drift detectors
│   └── monitoring.py
├── reports/                    # Generated JSON/HTML drift reports
│   └── drift_report.json
├── src/                        # Core machine learning pipelines & feature engineering
│   ├── model.py
│   └── preprocessing.py
├── streamlit_app/              # Real-time monitoring UI dashboard
│   └── app.py
├── tests/                      # Comprehensive test suite
│   ├── integration/
│   │   └── test_api.py
│   └── unit/
│       ├── test_data.py
│       └── test_model.py
├── docker-compose.yml          # Full stack containerization
├── Dockerfile
├── requirements.txt
└── README.md

🧠 Core Features & MLOps Highlights

    Real-Time Inference API (FastAPI):

        Receives multi-sensor time-series engine readings.

        Computes RUL instantly and logs every request asynchronously into PostgreSQL.

    Automated Continuous Monitoring (Apache Airflow + Evidently AI):

        Configured DAGs inspect production prediction logs (e.g., sliding window of the last 24 hours).

        Automatically benchmarks live data against the static reference dataset (train.csv).

        Evaluates feature distribution shifts and dataset-level drift flags.

    Closed-Loop Continuous Training (CT):

        If dataset_drift = True is detected, Airflow dynamically triggers the automated training pipeline to ingest fresh data and update the model weights.

    Instant Telemetry & Alerting (Telegram):

        Automatically dispatches real-time Telegram alerts upon detecting critical drift metrics or successful model retraining completions.

    Operational Intelligence Dashboard (Streamlit):

        Displays engine status, active alerts, RUL degradation curves over time, and live data health metrics in a single interface.

    Robust Testing Suite (Pytest):

        Fully covered with unit tests for data preprocessing and model inference, alongside integration tests for the REST API endpoints.

🚀 Getting Started & Installation
1. Clone the Repository
Bash

git clone https://github.com/ahmedhany-stack/sensors_project
cd rul-mlops-pipeline

2. Environment Variables

Create a .env file in the root directory based on .env.example:
مقتطف الرمز

POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=rul_db
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

3. Run with Docker Compose

Bring up the entire stack (API, PostgreSQL, Airflow, and Streamlit) with a single command:
Bash

docker-compose up --build

4. Run Tests

Verify the integrity of the system using pytest:
Bash

pytest tests/

📊 Dashboard Preview & Monitoring Loop

    FastAPI Docs: http://localhost:8000/docs

    Streamlit UI: http://localhost:8501

    Airflow Webserver: http://localhost:8080