import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str = os.path.join("models", "metrics.json")

class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, model_path: str, transformed_train_path: str):
        logger.info("Starting Model Evaluation with MLflow tracking...")
        try:
            model = joblib.load(model_path)
            train_df = pd.read_csv(transformed_train_path)

            X = train_df.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
            y = train_df['RUL']

            predictions = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, predictions))
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)

            metrics = {
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2_Score": float(r2)
            }

            logger.info(f"Evaluation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            # حفظ الـ Metrics محلياً في JSON
            os.makedirs(os.path.dirname(self.evaluation_config.metrics_file_path), exist_ok=True)
            with open(self.evaluation_config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            # تسجيل الـ Metrics في MLflow (ضمن الـ active run الحالية إن وجدت، أو فتح run جديدة)
            if mlflow.active_run() is None:
                mlflow.start_run(run_name="Model_Evaluation", nested=True)
            
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self.evaluation_config.metrics_file_path)

            logger.info("Model Evaluation completed and logged to MLflow successfully.")
            return metrics

        except Exception as e:
            logger.error("Error during Model Evaluation.")
            raise CustomException(e, sys)