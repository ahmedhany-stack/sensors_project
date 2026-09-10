import os
import sys
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.onnx
from xgboost import XGBRegressor
from dataclasses import dataclass
from onnxmltools import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType
from src.utils.logger import logger
from src.utils.exception import CustomException


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
trainer_cfg = config.get("model_trainer", {})
paths_cfg = trainer_cfg.get("paths", {})
mlflow_cfg = trainer_cfg.get("mlflow", {})
params_cfg = trainer_cfg.get("hyperparameters", {})
cols_cfg = trainer_cfg.get("columns", {})


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = paths_cfg.get("model_file", os.path.join("models", "xgb_rul_model.onnx"))


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
        self.experiment_name = mlflow_cfg.get("experiment_name", "RUL_Prediction")
        self.run_name = mlflow_cfg.get("run_name", "XGBoost_RUL_Training_ONNX")
        self.exclude_cols = cols_cfg.get("exclude_cols", ['unit_number', 'time_in_cycles', 'RUL'])
        
        # Hyperparameters
        self.n_estimators = params_cfg.get("n_estimators", 100)
        self.learning_rate = params_cfg.get("learning_rate", 0.05)
        self.max_depth = params_cfg.get("max_depth", 5)
        self.random_state = params_cfg.get("random_state", 42)

    def initiate_model_trainer(self, transformed_train_path: str):
        logger.info("Starting Model Training with MLflow tracking and ONNX export...")
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            
            if not os.path.exists(transformed_train_path):
                raise FileNotFoundError(f"Transformed training data not found at: {transformed_train_path}")

            train_df = pd.read_csv(transformed_train_path)
            logger.info(f"Loaded transformed training data with shape: {train_df.shape}")

            X_train = train_df.drop(columns=self.exclude_cols, errors='ignore')
            y_train = train_df['RUL']

            with mlflow.start_run(run_name=self.run_name):
                # تسجيل الـ Parameters في MLflow
                mlflow.log_param("n_estimators", self.n_estimators)
                mlflow.log_param("learning_rate", self.learning_rate)
                mlflow.log_param("max_depth", self.max_depth)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("model_format", "ONNX")

                model = XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )

                logger.info("Fitting XGBoost Regressor model...")
                X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
                
                model.fit(X_train_values, y_train_values)

                # -------------------------------------------------------------
                # 1. تحويل موديل XGBoost إلى صيغة ONNX
                # -------------------------------------------------------------
                logger.info("Converting XGBoost model to ONNX format...")
                num_features = X_train.shape[1]
                
                initial_type = [('float_input', FloatTensorType([None, num_features]))]
                onnx_model = convert_xgboost(model, initial_types=initial_type)

                # -------------------------------------------------------------
                # 2. حفظ ملف .onnx محلياً
                # -------------------------------------------------------------
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                # -------------------------------------------------------------
                # 3. تسجيل موديل الـ ONNX في MLflow
                # -------------------------------------------------------------
                mlflow.onnx.log_model(onnx_model, artifact_path="onnx_model")

                logger.info(f"ONNX Model trained and saved successfully at: {self.model_trainer_config.trained_model_file_path}")
                return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            logger.error("Error occurred during Model Training / ONNX Conversion.")
            raise CustomException(e, sys)