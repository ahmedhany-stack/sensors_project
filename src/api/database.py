import os
import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# تحميل المتغيرات من ملف .env
load_dotenv()

# قراءة رابط قاعدة البيانات من البيئة
DATABASE_URL = os.getenv("DATABASE_URL")

# في حالة عدم وجود المتغير لأي سبب، نقوم بتشكيله من المتغيرات المنفصلة
if not DATABASE_URL:
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "rul_db")
    DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

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
        for rec, pred in zip(records_data, raw_predictions):
            log_item = PredictionLog(
                unit_number=rec.get("unit_number"),
                time_in_cycles=rec.get("time_in_cycles"),
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