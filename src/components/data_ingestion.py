import os
import sys
import pandas as pd
import yaml
from dataclasses import dataclass
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.utils.logger import logger
from src.utils.exception import CustomException

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
ingestion_cfg = config.get("data_ingestion", {})
paths_cfg = ingestion_cfg.get("paths", {})
db_ingest_cfg = ingestion_cfg.get("database", {})
cols_cfg = ingestion_cfg.get("columns", {})


@dataclass
class DataIngestionConfig:
    raw_data_path: str = paths_cfg.get("raw_train", os.path.join("data", "raw", "train_FD001.txt"))
    test_raw_data_path: str = paths_cfg.get("raw_test", os.path.join("data", "raw", "test_FD001.txt"))
    ingested_train_path: str = paths_cfg.get("ingested_train", os.path.join("data", "processed", "train.csv"))
    ingested_test_path: str = paths_cfg.get("ingested_test", os.path.join("data", "processed", "test.csv"))


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", db_ingest_cfg.get("default_host", "localhost"))
        db_port = os.getenv("DB_PORT", str(db_ingest_cfg.get("default_port", "5432")))
        db_name = os.getenv("DB_NAME", db_ingest_cfg.get("default_db_name", "rul_db"))
        
        if not db_user or not db_password:
            raise ValueError("❌ Database credentials (DB_USER or DB_PASSWORD) are missing from environment variables!")

        self.database_url = os.getenv("DATABASE_URL", f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
        self.engine = create_engine(self.database_url)
        self.train_table = db_ingest_cfg.get("train_table", "historical_training_data")
        self.test_table = db_ingest_cfg.get("test_table", "historical_test_data")

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion process from/to Database...")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.ingested_train_path), exist_ok=True)
            
            # تجميع أسماء الأعمدة ديناميكياً من الـ Config
            base_cols = cols_cfg.get("base", ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'])
            num_sensors = cols_cfg.get("num_sensors", 21)
            columns = base_cols + [f's_{i}' for i in range(1, num_sensors + 1)]
            
            try:
                query_check = f"SELECT COUNT(*) FROM {self.train_table};"
                df_count = pd.read_sql(query_check, self.engine)
                db_has_data = df_count.iloc[0, 0] > 0
            except Exception:
                db_has_data = False

            if not db_has_data:
                logger.info("Database training table is empty. Loading raw text files and pushing to database (Bootstrap phase)...")
                
                df_train = pd.read_csv(self.ingestion_config.raw_data_path, sep=r'\s+', header=None, names=columns)
                df_test = pd.read_csv(self.ingestion_config.test_raw_data_path, sep=r'\s+', header=None, names=columns)
                
                df_train.to_sql(
                                self.train_table, 
                                self.engine, 
                                if_exists="replace", 
                                index=False, 
                                chunksize=10000, 
                            
                            )
                df_test.to_sql(
                                self.test_table, 
                                self.engine, 
                                if_exists="replace", 
                                index=False, 
                                chunksize=10000, 
                             
                            )
                logger.info("Successfully pushed raw train and test data into PostgreSQL tables.")
            
            logger.info("Fetching training and testing data from PostgreSQL database...")
            df_train = pd.read_sql(f"SELECT * FROM {self.train_table};", self.engine)
            df_test = pd.read_sql(f"SELECT * FROM {self.test_table};", self.engine)
            
            logger.info(f"Loaded train data shape from DB: {df_train.shape}, test data shape from DB: {df_test.shape}")
            
            df_train.to_csv(self.ingestion_config.ingested_train_path, index=False)
            df_test.to_csv(self.ingestion_config.ingested_test_path, index=False)
            
            logger.info("Data Ingestion from Database completed successfully.")
            return self.ingestion_config.ingested_train_path, self.ingestion_config.ingested_test_path

        except Exception as e:
            logger.error("Error occurred in Data Ingestion.")
            raise CustomException(e, sys)