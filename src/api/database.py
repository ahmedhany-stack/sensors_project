import os
import datetime
from dotenv import load_dotenv
import yaml
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# تحميل المتغيرات من ملف .env
load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
db_cfg = config.get("database", {})
defaults = db_cfg.get("defaults", {})
tables = db_cfg.get("table_names", {})
db_keys = db_cfg.get("keys", {})

# جلب بيانات الاتصال (المتغيرات الحساسة من .env مع الـ Fallback من config.yaml)
db_user = os.getenv("DB_USER", defaults.get("user", "airflow"))
db_password = os.getenv("DB_PASSWORD", "airflow")
db_host = os.getenv("DB_HOST", defaults.get("host", "127.0.0.1"))
db_port = os.getenv("DB_PORT", str(defaults.get("port", 5433)))
db_name = os.getenv("DB_NAME", defaults.get("name", "rul_db"))

# بناء رابط الاتصال
DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

print(f"--> Connecting to Database URL: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
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


# إنشاء الجدول تلقائياً إن لم يكن موجوداً
Base.metadata.create_all(bind=engine)


def save_predictions_to_db(records_data: list, raw_predictions: list):
    """دالة خلفية لتخزين الطلبات والتوقعات في PostgreSQL"""
    db = SessionLocal()
    try:
        logs = []
        unit_key = db_keys.get("unit_number", "unit_number")
        cycles_key = db_keys.get("time_in_cycles", "time_in_cycles")

        for rec, pred in zip(records_data, raw_predictions):
            log_item = PredictionLog(
                unit_number=rec.get(unit_key),
                time_in_cycles=rec.get(cycles_key),
                features=rec,
                predicted_rul=float(pred)
            )
            logs.append(log_item)
        db.add_all(logs)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to log to Postgres: {e}")
    finally:
        db.close()