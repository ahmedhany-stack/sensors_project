import os
import sys

# ⚡ الحل الجذري والوحيد المضمون لأيرفلو: إضافة جذر المشروع مباشرة للـ sys.path
AIRFLOW_ROOT = "/opt/airflow"
if AIRFLOW_ROOT not in sys.path:
    sys.path.insert(0, AIRFLOW_ROOT)

# التأكد برضه من مسار الـ dags أو مجلد المشروع الحالي لو بتشغله محلياً
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

# دلوقتي استورد براحتك من غير ما تشيل هم الـ ModuleNotFoundError نهائياً
from src.api.monitoring import drift_monitor
from src.pipelines.training_pipeline import TrainingPipeline

# إعداد الـ Logger الخاص بالـ DAG
logger = logging.getLogger("airflow.task")

# قراءة إعدادات الـ Telegram بأمان تام بدون حرقها في الكود
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# نتأكد إنهم موجودين فعلاً وإلا نوقف الكود
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("❌ Telegram credentials are missing from environment variables!")

def send_telegram_alert(message: str):
    """دالة لإرسال التنبيهات على تليجرام باستخدام متغيرات البيئة"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram Token or Chat ID is missing from environment variables!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully!")
        else:
            logger.error(f"Failed to send telegram alert: {response.text}")
    except Exception as e:
        logger.error(f"Error sending telegram alert: {e}")

# إعدادات الـ DAG الافتراضية
default_args = {
    "owner": "mlops_engine",
    "depends_on_past": False,
    "start_date": datetime(2026, 8, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def generate_and_check_drift():
    """دالة لسحب البيانات، فحص الـ Drift، وإعادة التدريب تلقائياً مع إرسال تنبيهات تليجرام"""
    logger.info("Starting Evidently Data Drift Monitoring & CT Task...")
    
    # ⚡ تم توجيه الرابط لقاعدة بيانات المشروع rul_db حيث يتواجد جدول prediction_logs فعلاً
    # (عدل اليوزر والباسورد لـ airflow:airflow لو دي البيانات اللي واخدها الـ container عندك)
    # استخدم بيانات الدخول الخاصة بـ docker-compose (يوزر والباسورد: airflow) ومع قاعدة بيانات rul_db
# استخدم بيانات الدخول الخاصة بـ docker-compose (يوزر والباسورد: airflow) ومع قاعدة بيانات rul_db
    db_url = "postgresql+psycopg2://airflow:airflow@postgres:5432/rul_db"    
    try:
        logger.info("Connecting to PostgreSQL database to fetch prediction logs...")
        engine = create_engine(db_url)
        
        # سحب بيانات آخر 24 ساعة
        query = """
            SELECT features 
            FROM prediction_logs 
            WHERE created_at >= NOW() - INTERVAL '1 DAY';
        """
        df_logs = pd.read_sql(query, engine)
        
        if df_logs.empty:
            logger.warning("No new prediction log data found in Postgres for the past 24 hours. Aborting drift analysis.")
            return

        logger.info(f"Successfully fetched {len(df_logs)} log records from database.")

        # فك الـ JSON المخزن في عمود features لجدول Pandas
        current_df = pd.DataFrame(df_logs["features"].tolist())
        logger.info(f"Parsed current features DataFrame with shape: {current_df.shape}")

        # تحميل عينة التدريب كـ Reference Data
        train_path = os.path.join("data", "processed", "train.csv")
        if os.path.exists(train_path):
            logger.info(f"Loading reference dataset from: {train_path}")
            ref_df = pd.read_csv(train_path)
            drift_monitor.set_reference_dataframe(ref_df)
        else:
            logger.warning(f"Reference dataset not found at path: {train_path}. Proceeding without setting new reference.")

        # 1. تشغيل تحليل الـ Drift
        logger.info("Running Evidently drift analysis pipeline...")
        drift_monitor.run_drift_analysis(current_data=current_df)
        
        # 2. قراءة تقرير الـ JSON الناتج لمعرفة هل حدث Drift أم لا بشكل آمن
        json_report_path = os.path.join("reports", "drift_report.json")
        drift_detected = False
        
        if os.path.exists(json_report_path):
            with open(json_report_path, "r") as f:
                report_data = json.load(f)
                
            # محاولة قراءة الـ Dataset Drift بطريقة آمنة ومتوافقة
            try:
                metrics = report_data.get("metrics", [])
                for metric in metrics:
                    if "DatasetDriftMetric" in metric.get("metric", ""):
                        drift_detected = metric.get("result", {}).get("dataset_drift", False)
                        break
                if not drift_detected and metrics:
                    drift_detected = metrics[0].get("result", {}).get("dataset_drift", False)
            except Exception as parse_err:
                logger.error(f"Error parsing drift JSON metrics structure: {parse_err}")
                drift_detected = False
        else:
            logger.warning("Drift JSON report path not found. Skipping automated trigger check.")

        logger.info(f"Data Drift Status: {drift_detected}")

        # 3. الشرط الذكي لإعادة التدريب (Continuous Training) والتنبيه
        if drift_detected:
            logger.warning("🚨 Data Drift detected! Triggering automated model retraining pipeline...")
            
            # استدعاء بايبلاين التدريب بشكل صحيح عبر Instance
            trainer_pipeline = TrainingPipeline()
            trainer_pipeline.run_pipeline()
            
            logger.info("Model retraining pipeline completed successfully after detecting drift!")
            
            # إرسال تنبيه نجاح الريترينينج على تليجرام
            alert_msg = (
                "🚨 *Alert: Data Drift Detected & Retrained!*\n\n"
                "📊 Evidently detected a significant dataset drift in the last 24 hours.\n"
                "🔄 Automatic model retraining pipeline (`TrainingPipeline`) has been executed successfully."
            )
            send_telegram_alert(alert_msg)
            
        else:
            logger.info("No data drift detected. Model is stable; retraining skipped.")

    except Exception as e:
        error_msg = f"❌ *Error in RUL MLOps Pipeline!*\n\nTask failed with exception: `{str(e)}`"
        logger.error(f"Failed to execute Monitoring and Retraining Task. Error: {str(e)}", exc_info=True)
        
        # إرسال تنبيه على تليجرام في حالة حدوث إيرور
        send_telegram_alert(error_msg)
        raise e


# تعريف الـ DAG
with DAG(
    "rul_data_drift_monitoring",
    default_args=default_args,
    description="Automated Data Drift monitoring with conditional model retraining and secure telegram alerts",
    schedule_interval="@daily",
    catchup=False,
) as dag:

    run_monitoring_task = PythonOperator(
        task_id="run_evidently_drift_analysis",
        python_callable=generate_and_check_drift,
    )