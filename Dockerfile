FROM apache/airflow:2.7.1

USER root

# تثبيت أدوات النظام الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# تثبيت المكتبات الأساسية بنطاقات آمنة تمنع أي تضارب
RUN pip install --no-cache-dir \
    "numpy>=1.22,<2.0" \
    "pandas>=2.0.0,<2.2.0" \
    scikit-learn \
    xgboost \
    mlflow \
    "evidently==0.4.34" \
    plotly \
    "pydantic>=2.0" \
    fastapi \
    PyYAML \
    requests \
    dvc \
    joblib \
    dill \
    "scipy>=1.9.0,<1.11.0" \
    statsmodels \
    nltk \
    psycopg2-binary