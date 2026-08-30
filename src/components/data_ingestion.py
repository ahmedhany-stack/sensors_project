import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "train_FD001.txt")
    test_raw_data_path: str = os.path.join("data", "raw", "test_FD001.txt")
    ingested_train_path: str = os.path.join("data", "processed", "train.csv")
    ingested_test_path: str = os.path.join("data", "processed", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion process...")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.ingested_train_path), exist_ok=True)
            
            columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
            
            df_train = pd.read_csv(self.ingestion_config.raw_data_path, sep=r'\s+', header=None, names=columns)
            df_test = pd.read_csv(self.ingestion_config.test_raw_data_path, sep=r'\s+', header=None, names=columns)
            
            logger.info(f"Loaded train data shape: {df_train.shape}, test data shape: {df_test.shape}")
            
            df_train.to_csv(self.ingestion_config.ingested_train_path, index=False)
            df_test.to_csv(self.ingestion_config.ingested_test_path, index=False)
            
            logger.info("Data Ingestion completed successfully.")
            return self.ingestion_config.ingested_train_path, self.ingestion_config.ingested_test_path

        except Exception as e:
            logger.error("Error occurred in Data Ingestion.")
            raise CustomException(e, sys)


