import os
import sys
import yaml
import pandas as pd
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
val_cfg = config.get("data_validation", {})
paths_cfg = val_cfg.get("paths", {})
schema_cfg = val_cfg.get("schema", {})


@dataclass
class DataValidationConfig:
    status_file_path: str = paths_cfg.get(
        "status_file", os.path.join("data", "processed", "data_validation_status.txt")
    )


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()
        
        # قراءة الأعمدة المتوقعة ديناميكياً من ملف الـ Config
        base_cols = schema_cfg.get(
            "base_columns", ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
        )
        num_sensors = schema_cfg.get("num_sensors", 21)
        self.expected_columns = base_cols + [f's_{i}' for i in range(1, num_sensors + 1)]

    def validate_all_columns(self, train_path: str) -> bool:
        """
        دالة للتحقق من صحة بيانات التدريب:
        1. التأكد من عدم وجود قيم مفقودة (Missing Values).
        2. التأكد من وجود جميع الأعمدة المتوقعة (Schema Validation).
        """
        logger.info("Starting Data Validation process...")
        try:
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

            # كتابة حالة الفحص في ملف الـ Status
            os.makedirs(os.path.dirname(self.validation_config.status_file_path), exist_ok=True)
            with open(self.validation_config.status_file_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data Validation completed. Overall Status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error("Error occurred during Data Validation.")
            raise CustomException(e, sys)