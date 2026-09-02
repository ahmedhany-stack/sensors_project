🛠️ Tech Stack & Advanced Components

    Core Language: Python, Pandas, NumPy, Scikit-Learn

    Machine Learning: XGBoost Regressor

    MLflow & Experiment Tracking: Parameter logging, metrics registry, and model artifact management

    Workflow Orchestration: Apache Airflow (Automated Pipelines, DAGs, and task scheduling)

    Drift Detection & Monitoring: Evidently AI & Prometheus Metrics

    Alerting System: Discord Webhooks & Telegram Bot integration for real-time notifications

    API Framework: FastAPI, Pydantic v2, and SQLAlchemy ORM

    Database Layer: PostgreSQL 13 (Centralized prediction logging and metrics storage)

    Containerization: Docker & Docker Compose

    Testing & Quality Assurance: Pytest (100% Unit & Integration Test Coverage)

📁 Project Structure
Plaintext

sensors_project/
├── .github/workflows/       # CI/CD GitHub Actions
├── airflow/dags/            # Airflow DAGs for automated retraining pipelines
├── data/                    # Raw, Processed, and Feature datasets
├── models/                  # Saved models, Scalers, and Metrics (JSON)
├── src/
│   ├── api/                 # FastAPI application and prediction endpoints
│   ├── components/          # Data Validation, Transformation, Training, Evaluation
│   ├── monitoring/          # Evidently AI drift detection scripts
│   └── utils/               # Custom logger, exception handlers, and webhook routers
├── tests/
│   ├── unit/                # Unit tests for data processing and models (Pytest)
│   └── integration/         # API endpoint integration tests
├── Dockerfile               # Production container config
├── docker-compose.yaml      # Multi-container orchestration (Airflow + PostgreSQL)
├── requirements.txt         # Project dependencies
└── pyproject.toml           # Pytest and project configuration

🛡️ Engineering Challenges, Troubleshooting & Real-World Solutions

Building a production-grade, multi-service MLOps system involves bridging local environments with containerized orchestrators. During the development and integration of this pipeline, several core architectural bottlenecks and system-level challenges were encountered and successfully resolved:
1. Python Environment & Dependency Resolution (ModuleNotFoundError)

    The Challenge: When initializing the FastAPI application or running components locally, Python threw ModuleNotFoundError and Could not import module exceptions due to incorrect sys.path resolution and relative package importing structures.

    The Solution: Standardized package execution by prefixing runtime commands with Python's module runner (python -m uvicorn src.api.app:app) and properly setting environment variables ($env:PYTHONPATH=".") to ensure absolute root-level module visibility across all local execution contexts.

2. Database Provisioning in Isolated Containers (Undefined Database rul_db)

    The Challenge: Airflow and the ML prediction logger required a centralized relational database (rul_db), which did not exist by default inside the initialized PostgreSQL 13 Docker container (sensors_project-postgres-1), causing connection refusal errors.

    The Solution: Intervened directly at the container level using docker exec with the PostgreSQL interactive terminal to provision the target database programmatically:
    Bash

    docker exec -it sensors_project-postgres-1 psql -U airflow -d airflow -c "CREATE DATABASE rul_db;"

3. Cross-Service Table Synchronization (UndefinedTable: prediction_logs)

    The Challenge: Even after database creation, Airflow DAGs executing drift analysis failed with psycopg2.errors.UndefinedTable because the prediction_logs table had not yet been materialized inside the container's rul_db instance.

    The Solution: Harmonized connection strings and unified user credentials (airflow:airflow) across both local .env configurations and Docker environment variables. We then explicitly executed table initialization scripts inside the container:
    Bash

    docker exec -it sensors_project-postgres-1 psql -U airflow -d rul_db -c "CREATE TABLE IF NOT EXISTS prediction_logs (id SERIAL PRIMARY KEY, features JSONB, prediction FLOAT, created_at TIMESTAMP DEFAULT NOW());"

4. Bridging Local API Endpoints with Containerized Orchestration

    The Challenge: A fundamental architectural hurdle arose when the FastAPI inference service ran locally (localhost:8000) while Apache Airflow executed inside isolated Docker containers (sensors_project-airflow-scheduler-1), causing disparity in data ingestion streams and log consumption layers.

    The Solution: Configured mapped Docker volume mounts (volumes: - .:/opt/airflow) and exposed container port 5432 to the host machine. This allowed local prediction requests sent to FastAPI to write directly into the shared Docker PostgreSQL volume, making logs instantly accessible to Airflow monitoring DAGs for seamless Evidently AI drift evaluation.

🔄 Automated Retraining & Drift Monitoring (The Core Magic)

    Drift Detection: Evidently AI continuously compares incoming production prediction logs stored in PostgreSQL against the baseline training dataset reference features.

    Alerting: If feature distributions or performance metrics drift beyond acceptable thresholds, critical visual alerts and reports are instantly dispatched via Discord Webhooks and Telegram Bots.

    Orchestration: Apache Airflow captures the trigger, initiates the automated retraining DAG, executes data processing steps, fits a fresh XGBoost model, logs all parameters and metrics to MLflow, and seamlessly registers the production model.

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
    .venv\\Scripts\\Activate  # On Windows PowerShell

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Run the FastAPI server:
    Bash

    python -m uvicorn src.api.app:app --reload

🐳 Docker Deployment

To build and run the orchestration services via Docker Compose:
Bash

docker compose up -d --build

🎯 Author

Ahmed Hany Sallam

Aspiring Machine Learning & MLOps Engineer
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)
print("README.md generated successfully!")

