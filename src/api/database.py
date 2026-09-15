import os
import datetime
import logging
from dotenv import load_dotenv
import yaml
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# تحميل المتغيرات من ملف .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_app")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# تحميل الإعدادات
config = load_config()
db_cfg = config.get("database", {})
defaults = db_cfg.get("defaults", {})
tables = db_cfg.get("table_names", {})
db_keys = db_cfg.get("keys", {})

# جلب بيانات الاتصال
db_user = os.getenv("DB_USER", defaults.get("user", "airflow"))
db_password = os.getenv("DB_PASSWORD", "airflow")
db_host = os.getenv("DB_HOST", defaults.get("host", "127.0.0.1"))
db_port = os.getenv("DB_PORT", str(defaults.get("port", 5433)))
db_name = os.getenv("DB_NAME", defaults.get("name", "rul_db"))

DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# إعداد الـ Connection Pool المطور للضغط العالي
engine = create_engine(
    DATABASE_URL,
    pool_size=100,      
    max_overflow=150,    
    pool_timeout=30,    
    pool_recycle=1800   
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = tables.get("prediction_logs", "prediction_logs")

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    unit_number = Column(Integer)
    time_in_cycles = Column(Integer)
    features = Column(JSON)
    predicted_rul = Column(Float)

def init_db():
    """تُستدعى هذه الدالة عند بدء تشغيل التطبيق (Startup) لإنشاء الجداول"""
    print(f"--> Connecting to Database URL: {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)

def save_predictions_to_db(records_data: list, raw_predictions: list):
    """الدالة القديمة (للتوافقية إن احتجتها)"""
    db = SessionLocal()
    try:
        unit_key = db_keys.get("unit_number", "unit_number")
        cycles_key = db_keys.get("time_in_cycles", "time_in_cycles")

        logs = [
            PredictionLog(
                unit_number=rec.get(unit_key),
                time_in_cycles=rec.get(cycles_key),
                features=rec,
                predicted_rul=float(pred)
            )
            for rec, pred in zip(records_data, raw_predictions)
        ]
        db.bulk_save_objects(logs)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log to Postgres in bulk: {e}")
    finally:
        db.close()

def save_predictions_bulk_to_db(batch_items: list):
    """
    تنفيذ Bulk Insert فائق السرعة لقائمة التنبؤات القادمة من الـ Queue Worker باستخدام PredictionLog
    """
    try:
        with Session(engine) as session:
            objects = [
                PredictionLog(
                    unit_number=item["unit_number"],
                    time_in_cycles=item["time_in_cycles"],
                    features=item.get("input_data", {}),
                    predicted_rul=float(item["prediction"])
                )
                for item in batch_items
            ]
            session.bulk_save_objects(objects)
            session.commit()
    except Exception as e:
        logger.error(f"Database bulk insert error: {e}")
        raise e