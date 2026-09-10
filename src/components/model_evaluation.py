import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import mlflow
import onnxruntime as ort
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger
from src.utils.exception import CustomException


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
eval_cfg = config.get("model_evaluation", {})
paths_cfg = eval_cfg.get("paths", {})
mlflow_cfg = eval_cfg.get("mlflow", {})
onnx_cfg = eval_cfg.get("onnx", {})
cols_cfg = eval_cfg.get("columns", {})


@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str = paths_cfg.get("metrics_file", os.path.join("models", "metrics.json"))


class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()
        self.run_name = mlflow_cfg.get("run_name", "Model_Evaluation_ONNX")
        self.providers = onnx_cfg.get("providers", ["CPUExecutionProvider"])
        self.exclude_cols = cols_cfg.get("exclude_cols", ['unit_number', 'time_in_cycles', 'RUL'])
        self.clip_upper = config.get("data_transformation", {}).get("rul", {}).get("clip_upper", 125)

        # البحث عن المسار في Config، وإذا لم يجده يتراجع للمسار الافتراضي
        self.y_column_path = (
            paths_cfg.get("y_column")
            or config.get("data_transformation", {}).get("paths", {}).get("y_column")
            or os.path.join("data", "raw", "RUL_FD001.txt")
        )

    def initiate_model_evaluation(self, model_path: str, evaluation_data_path: str):
        logger.info("Starting ONNX Model Evaluation process with MLflow tracking...")
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained ONNX model not found at: {model_path}")
            if not os.path.exists(evaluation_data_path):
                raise FileNotFoundError(f"Evaluation dataset not found at: {evaluation_data_path}")
            if not os.path.exists(self.y_column_path):
                raise FileNotFoundError(f"Ground truth RUL file not found at: {self.y_column_path}")

            eval_df = pd.read_csv(evaluation_data_path)
            logger.info(f"Loaded evaluation dataset with shape: {eval_df.shape}")

            # 1. تجهيز الـ X وحساب الـ Y بدقة
            X = eval_df.drop(columns=self.exclude_cols, errors='ignore')
            max_cycles = eval_df.groupby('unit_number')['time_in_cycles'].transform('max')
            
            # قراءة ملف الـ RUL الحقيقي
            rul_ground_truth = pd.read_csv(self.y_column_path, header=None, names=['true_rul'])
            rul_ground_truth['unit_number'] = rul_ground_truth.index + 1
            
            eval_df = eval_df.merge(rul_ground_truth, on='unit_number', how='left')
            eval_df['RUL'] = eval_df['true_rul'] + (max_cycles - eval_df['time_in_cycles'])
            y = eval_df['RUL'].clip(upper=self.clip_upper)

            # 2. تحميل موديل ONNX
            logger.info(f"Loading ONNX model session from: {model_path}")
            session = ort.InferenceSession(model_path, providers=self.providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # 3. تحويل البيانات وتجهيز الـ Inference
            X_input = X.values.astype(np.float32)
            predictions_raw = session.run([output_name], {input_name: X_input})[0]
            predictions = predictions_raw.flatten()

            # 4. حساب الـ Metrics
            rmse = np.sqrt(mean_squared_error(y, predictions))
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)

            metrics = {
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2_Score": float(r2)
            }

            logger.info(f"ONNX Evaluation Metrics -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            # 5. حفظ وحفظ الـ Metrics في MLflow
            os.makedirs(os.path.dirname(self.evaluation_config.metrics_file_path), exist_ok=True)
            with open(self.evaluation_config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            if mlflow.active_run() is None:
                mlflow.start_run(run_name=self.run_name, nested=True)
            
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self.evaluation_config.metrics_file_path)

            logger.info("ONNX Model Evaluation completed and logged to MLflow successfully.")
            return metrics

        except Exception as e:
            logger.error("Error occurred during ONNX Model Evaluation.")
            raise CustomException(e, sys)