import os
import sys
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

@dataclass
class ModelTrainerConfig:
    # امتداد الملف .onnx
    trained_model_file_path: str = os.path.join("models", "xgb_rul_model.onnx")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, transformed_train_path: str):
        logger.info("Starting Model Training with MLflow tracking and ONNX export...")
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("RUL_Prediction")
            
            if not os.path.exists(transformed_train_path):
                raise FileNotFoundError(f"Transformed training data not found at: {transformed_train_path}")

            train_df = pd.read_csv(transformed_train_path)
            logger.info(f"Loaded transformed training data with shape: {train_df.shape}")

            X_train = train_df.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
            y_train = train_df['RUL']

            # Hyperparameters
            n_estimators = 100
            learning_rate = 0.05
            max_depth = 5
            random_state = 42

            with mlflow.start_run(run_name="XGBoost_RUL_Training_ONNX"):
                # تسجيل الـ Parameters في MLflow
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("model_format", "ONNX")

                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )

                logger.info("Fitting XGBoost Regressor model...")
                # تحويل البيانات إلى NumPy Array لتفادي حفظ أسماء الأعمدة داخل الموديل
                X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
                
                model.fit(X_train_values, y_train_values)

                # -------------------------------------------------------------
                # 1. تحويل موديل XGBoost إلى صيغة ONNX
                # -------------------------------------------------------------
                logger.info("Converting XGBoost model to ONNX format...")
                num_features = X_train.shape[1]
                
                # تعريف الـ Input Tensor بالأبعاد المطلوبة
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