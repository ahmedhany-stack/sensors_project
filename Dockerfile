FROM apache/airflow:2.7.1

USER root

# تثبيت أدوات النظام مع إمكانية تجاوز انتهاء صلاحية مستودعات Debian Oldstable
RUN apt-get update --allow-releaseinfo-change || true && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# تثبيت المكتبات وتضمين مكتبات ONNX المطلوبة
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
    psycopg2-binary \
    onnxmltools \
    onnxconverter_common \
    skl2onnx \
    onnxruntime