import os
import sys
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "xgb_rul_model.joblib")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, transformed_train_path: str):
        logger.info("Starting Model Training with MLflow tracking...")
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

            # تفعيل الـ MLflow tracking run
            with mlflow.start_run(run_name="XGBoost_RUL_Training"):
                # تسجيل الـ Parameters في MLflow
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("random_state", random_state)

                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state
                )

                logger.info("Fitting XGBoost Regressor model...")
                model.fit(X_train, y_train)

                # حفظ الموديل محلياً
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                joblib.dump(model, self.model_trainer_config.trained_model_file_path)

                # تسجيل الموديل في MLflow
                mlflow.xgboost.log_model(model, "model")

                logger.info(f"Model trained and saved successfully at: {self.model_trainer_config.trained_model_file_path}")
                return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            logger.error("Error occurred during Model Training.")
            raise CustomException(e, sys)