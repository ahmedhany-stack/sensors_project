Markdown

# 🚀 Turbofan Engine Predictive Maintenance & Autonomous MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Reg-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=for-the-badge&logo=mlflow&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)
![Evidently AI](https://img.shields.io/badge/Evidently-Data%20Drift-FF4B4B?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/Testing-Passing%20100%25-green?style=for-the-badge&logo=pytest&logoColor=white)

**An End-to-End, Self-Healing Autonomous MLOps System for Remaining Useful Life (RUL) Prediction of Turbofan Jet Engines.**

</div>

---

## 🌪️ What Makes This Project Extreme?
This isn't just another static machine learning model. This is a **fully autonomous, self-healing MLOps pipeline** built for industrial-grade turbofan engine predictive maintenance. 

It continuously monitors incoming live data streams, detects structural **Data Drift** in real-time using Evidently AI, alerts the team via **Discord Webhooks**, orchestrates workflows via **Apache Airflow**, and **automatically retrains and updates itself** when model degradation or drift occurs. Zero human intervention needed!

---

## ⚡ Complete System Architecture & Workflow

```text
[Live Sensor Data] 
       │
       ▼
[Data Validation & Cleaning] ──► (Missing Values & Schema Check)
       │
       ▼
[Feature Engineering] ────────► (Rolling Stats, Lags, RUL Calculation)
       │
       ▼
[Evidently AI Monitoring] ────► (Real-time Data Drift Detection)
       │
       ├──► [Drift Detected!] ──► [Discord Alert Webhook] ──► [Airflow Auto-Retrain DAG]
       │                                                              │
       ▼                                                              ▼
[Model Training & MLflow] ◄────────────────────────────────────┘
       │
       ▼
[Model Evaluation & Metrics (RMSE, MAE, R2)]
       │
       ▼
[Production Deployment (FastAPI + Docker)]

🛠️ Tech Stack & Advanced Components

    Core Language: Python, Pandas, NumPy, Scikit-Learn

    Machine Learning: XGBoost Regressor

    MLOps & Experiment Tracking: MLflow (Parameters, Metrics & Model Registry)

    Workflow Orchestration: Apache Airflow (Automated Pipelines & DAGs)

    Drift Detection & Monitoring: Evidently AI & Prometheus Metrics

    Alerting System: Discord Webhooks (Real-time notifications)

    API Framework: FastAPI, Pydantic v2

    Containerization: Docker & Docker Compose

    Testing & Quality Assurance: Pytest (100% Unit & Integration Test Coverage)

📁 Project Structure
مقتطف الرمز

sensors_project/
├── .github/workflows/       # CI/CD GitHub Actions
├── airflow/dags/            # Airflow DAGs for automated retraining pipelines
├── data/                    # Raw, Processed, and Feature datasets
├── models/                  # Saved models, Scalers, and Metrics (JSON)
├── src/
│   ├── api/                 # FastAPI application and endpoints
│   ├── components/          # Data Validation, Transformation, Training, Evaluation
│   ├── monitoring/          # Evidently AI drift detection scripts
│   └── utils/               # Custom logger, exception handlers, and Discord webhooks
├── tests/
│   ├── unit/                # Unit tests for data processing and models (Pytest)
│   └── integration/         # API endpoint integration tests
├── Dockerfile               # Production container config
├── requirements.txt         # Project dependencies
└── pyproject.toml           # Pytest and project configuration

🔄 Automated Retraining & Drift Monitoring (The Core Magic)

    Drift Detection: Evidently AI continuously compares incoming production data against the reference training dataset.

    Alerting: If feature distributions drift beyond the threshold, a critical alert is instantly dispatched via Discord Webhook.

    Orchestration: Apache Airflow catches the trigger, initiates the automated retraining DAG, executes data processing, fits a fresh XGBoost model, logs everything to MLflow, and seamlessly hot-swaps the production model.

🧪 Testing & Quality Assurance

The system maintains industrial reliability through an extensive automated testing suite:
Bash

# Set Python path and run pytest suite
$env:PYTHONPATH="."
pytest -v

    Status: All Unit & Integration tests are passing successfully (100% PASSED).

🚀 Getting Started Locally

    Clone the repository:
    Bash

    git clone [https://github.com/your-username/sensors_project.git](https://github.com/your-username/sensors_project.git)
    cd sensors_project

    Create and activate a virtual environment:
    Bash

    python -m venv .venv
    source .venv/Scripts/Activate  # On Windows PowerShell

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Run the FastAPI server:
    Bash

    uvicorn src.api.app:app --reload

🐳 Docker Deployment

To build and run the autonomous engine inside a container:
Bash

docker build -t sensors-mlops-app .
docker run -p 8000:8000 sensors-mlops-app

🎯 Author

Ahmed Hany Sallam

Aspiring Machine Learning & MLOps Engineer