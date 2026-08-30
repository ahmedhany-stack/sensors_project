import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class DataValidationConfig:
    status_file_path: str = os.path.join("data", "processed", "data_validation_status.txt")

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_all_columns(self, train_path: str) -> bool:
        logger.info("Starting Data Validation process...")
        try:
            data = pd.read_csv(train_path)
            validation_status = True
            
            if data.isnull().sum().sum() > 0:
                validation_status = False
                logger.warning("Data Validation Warning: Missing values detected!")

            os.makedirs(os.path.dirname(self.validation_config.status_file_path), exist_ok=True)
            with open(self.validation_config.status_file_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data Validation completed. Status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error("Error during Data Validation.")
            raise CustomException(e, sys)