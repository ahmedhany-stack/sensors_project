# استخدام نسخة Python خفيفة ومستقرة

FROM apache/airflow:2.7.1-python3.10

USER airflow
RUN pip install --no-cache-dir "evidently<0.5.0" psycopg2-binary

