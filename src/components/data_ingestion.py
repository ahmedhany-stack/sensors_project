import os
import sys
import pandas as pd
from dataclasses import dataclass
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.utils.logger import logger
from src.utils.exception import CustomException

load_dotenv()

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "train_FD001.txt")
    test_raw_data_path: str = os.path.join("data", "raw", "test_FD001.txt")
    # تم الاحتفاظ بالمسارات كـ Artifacts مؤقتة لو باقي البايبلاين بيحتاجها، لكن المصدر الأساسي اصبح Database
    ingested_train_path: str = os.path.join("data", "processed", "train.csv")
    ingested_test_path: str = os.path.join("data", "processed", "test.csv")

class DataIngestion:
    def __init__(self):
            self.ingestion_config = DataIngestionConfig()
        
    # قراءة متغيرات الاتصال من ملف الـ .env حصرياً ودون أي قيم حساسة في الكود
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST", "localhost")  # الـ localhost هنا آمن لأنه مش بيانات حساسة
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "rul_db")
            
            # التأكد من وجود المتغيرات الأساسية لمنع الأخطاء المفاجئة
            if not db_user or not db_password:
                raise ValueError("❌ Database credentials (DB_USER or DB_PASSWORD) are missing from environment variables!")

            self.database_url = os.getenv("DATABASE_URL", f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
            self.engine = create_engine(self.database_url)

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion process from/to Database...")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.ingested_train_path), exist_ok=True)
            
            columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
            
            # التحقق مما إذا كانت البيانات موجودة مسبقاً في الداتا بيز أم لا (للبدء أول مرة)
            try:
                query_check = "SELECT COUNT(*) FROM historical_training_data;"
                df_count = pd.read_sql(query_check, self.engine)
                db_has_data = df_count.iloc[0, 0] > 0
            except Exception:
                db_has_data = False # الجدول لسه مش منشأ

            if not db_has_data:
                logger.info("Database training table is empty. Loading raw text files and pushing to database (Bootstrap phase)...")
                
                # قراءة الملفات الخام
                df_train = pd.read_csv(self.ingestion_config.raw_data_path, sep=r'\s+', header=None, names=columns)
                df_test = pd.read_csv(self.ingestion_config.test_raw_data_path, sep=r'\s+', header=None, names=columns)
                
                # حفظها في الداتا بيز كمرجع أساسي (Historical Data)
                df_train.to_sql("historical_training_data", self.engine, if_exists="replace", index=False)
                df_test.to_sql("historical_test_data", self.engine, if_exists="replace", index=False)
                logger.info("Successfully pushed raw train and test data into PostgreSQL tables.")
            
            # سحب البيانات المطلوبة للتدريب مباشرة من الداتا بيز (وليست من ملفات الـ CSV)
            logger.info("Fetching training and testing data from PostgreSQL database...")
            df_train = pd.read_sql("SELECT * FROM historical_training_data;", self.engine)
            df_test = pd.read_sql("SELECT * FROM historical_test_data;", self.engine)
            
            logger.info(f"Loaded train data shape from DB: {df_train.shape}, test data shape from DB: {df_test.shape}")
            
            # (اختياري) حفظها كملفات CSV مؤقتة لو باقي البايبلاين (Transformation/Training) متصمم يقرأ ملفات
            # لكن الأفضل لاحقاً تخلي باقي الكومبوننتس تقرأ Dataframe مباشرة.
            df_train.to_csv(self.ingestion_config.ingested_train_path, index=False)
            df_test.to_csv(self.ingestion_config.ingested_test_path, index=False)
            
            logger.info("Data Ingestion from Database completed successfully.")
            return self.ingestion_config.ingested_train_path, self.ingested_test_path

        except Exception as e:
            logger.error("Error occurred in Data Ingestion.")
            raise CustomException(e, sys)