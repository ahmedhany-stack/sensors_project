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
        # تعريف الأعمدة الأساسية المتوقعة لمشروع الـ RUL (Turbofan Engine)
        self.expected_columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

    def validate_all_columns(self, train_path: str) -> bool:
        """
        دالة للتحقق من صحة بيانات التدريب:
        1. التأكد من عدم وجود قيم مفقودة (Missing Values).
        2. التأكد من وجود جميع الأعمدة المتوقعة (Schema Validation).
        """
        logger.info("Starting Data Validation process...")
        try:
            # قراءة البيانات (سواء جاية من الـ CSV المؤقت اللي جهزه الـ Ingestion من الـ DB، أو مسار مباشر)
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data path not found at: {train_path}")
                
            data = pd.read_csv(train_path)
            validation_status = True
            missing_cols = []

            # 1. التحقق من وجود جميع الأعمدة الأساسية (Schema Validation)
            for col in self.expected_columns:
                if col not in data.columns:
                    validation_status = False
                    missing_cols.append(col)
            
            if missing_cols:
                logger.error(f"Data Validation Error: Missing expected columns: {missing_cols}")
            else:
                logger.info("Schema validation passed: All expected columns are present.")

            # 2. التحقق من وجود قيم مفقودة (Null Values Check)
            null_counts = data.isnull().sum().sum()
            if null_counts > 0:
                validation_status = False
                logger.warning(f"Data Validation Warning: Found {null_counts} missing values in the dataset!")
            else:
                logger.info("Null values validation passed: No missing values found.")

            # كتابة حالة الفحص في ملف الـ Status (عشان الـ Airflow أو البايبلاين يقراها)
            os.makedirs(os.path.dirname(self.validation_config.status_file_path), exist_ok=True)
            with open(self.validation_config.status_file_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data Validation completed. Overall Status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error("Error occurred during Data Validation.")
            raise CustomException(e, sys)