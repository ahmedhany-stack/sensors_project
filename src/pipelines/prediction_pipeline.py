import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات من config.yaml
config = load_config()

# استخراج الإعدادات المخصصة لكل مرحلة
trainer_cfg = config.get("model_trainer", {})
transform_cfg = config.get("data_transformation", {})
eval_cfg = config.get("model_evaluation", {})
ingest_cfg = config.get("data_ingestion", {})

# قراءة مسار الـ Full ONNX Pipeline المدمجة من الـ Config
combined_pipeline_path_cfg = transform_cfg.get("paths", {}).get(
    "output_onnx_path", os.path.join("models", "full_rul_pipeline.onnx")
)

features_path_cfg = transform_cfg.get("paths", {}).get(
    "features_json", os.path.join("models", "features.json")
)
test_file_path_cfg = ingest_cfg.get("paths", {}).get(
    "ingested_test", os.path.join("data", "processed", "test.csv")
)

onnx_providers_cfg = eval_cfg.get("onnx", {}).get("providers", ["CPUExecutionProvider"])
constant_cols_cfg = transform_cfg.get("columns", {}).get(
    "constant_cols", ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
)
fe_cfg = transform_cfg.get("feature_engineering", {})


@dataclass
class PredictionPipelineConfig:
    full_pipeline_path: str = combined_pipeline_path_cfg
    features_path: str = features_path_cfg


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.providers = onnx_providers_cfg
        self.constant_cols = constant_cols_cfg
        self.rolling_window = fe_cfg.get("rolling_window", 10)
        self.rolling_min_periods = fe_cfg.get("rolling_min_periods", 1)
        self.lag_step = fe_cfg.get("lag_step", 1)
        self._load_artifacts()

    def _load_artifacts(self):
        """تحميل الـ ONNX Runtime Session للملف المدمج الكامل وقائمة الـ Features"""
        try:
            logger.info("Loading inference artifacts (Full ONNX Pipeline, Features list)...")
            
            if not os.path.exists(self.config.full_pipeline_path):
                raise FileNotFoundError(
                    f"Combined ONNX Pipeline file not found at {self.config.full_pipeline_path}"
                )
            if not os.path.exists(self.config.features_path):
                raise FileNotFoundError(
                    f"Features JSON file not found at {self.config.features_path}"
                )

            # -------------------------------------------------------------
            # إنشاء Single ONNX Session للـ Full Pipeline (Scaler + Model)
            # -------------------------------------------------------------
            self.pipeline_session = ort.InferenceSession(
                self.config.full_pipeline_path, 
                providers=self.providers
            )
            self.input_name = self.pipeline_session.get_inputs()[0].name
            self.output_name = self.pipeline_session.get_outputs()[0].name

            # تحميل قائمة الـ Features المسجلة أثناء التدريب
            with open(self.config.features_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)

            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error("Failed to load artifacts in PredictionPipeline.")
            raise CustomException(e, sys)

    def _preprocess_input_data(self, df: pd.DataFrame) -> np.ndarray:
        """تطبيق Feature Engineering فقط وإرجاع NumPy Array جاهزة للـ Full ONNX Graph"""
        try:
            df = df.copy()

            # 1. إزالة الأعمدة الثابتة
            df.drop(columns=[c for c in self.constant_cols if c in df.columns], inplace=True)

            # 2. تطبيق الـ Rolling والـ Lag بناءً على معاملات الـ Config
            sensor_cols = [c for c in df.columns if c.startswith('s_') or c.startswith('setting_')]
            
            for col in sensor_cols:
                df[f'{col}_roll_mean'] = df.groupby('unit_number')[col].transform(
                    lambda x: x.rolling(self.rolling_window, min_periods=self.rolling_min_periods).mean()
                )
                df[f'{col}_roll_std'] = df.groupby('unit_number')[col].transform(
                    lambda x: x.rolling(self.rolling_window, min_periods=self.rolling_min_periods).std()
                ).fillna(0)
                df[f'{col}_lag_{self.lag_step}'] = df.groupby('unit_number')[col].shift(self.lag_step)
                df[f'{col}_lag_{self.lag_step}'] = df.groupby('unit_number')[f'{col}_lag_{self.lag_step}'].bfill()

            # 3. التأكد من تطابق الأعمدة مع features.json
            missing_cols = set(self.feature_names) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required feature columns in input data: {missing_cols}")

            # ترتيب الأعمدة وتحويلها إلى float32
            X_raw = df[self.feature_names].values.astype(np.float32)

            return X_raw

        except Exception as e:
            logger.error("Error during input data preprocessing in PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """تشغيل التوقع مباشرة باستخدام Full ONNX Pipeline Session"""
        try:
            logger.info(f"Starting ONNX prediction for input data of shape {input_df.shape}...")
            
            # تجهيز البيانات الخام فقط (التشفيير والـ Scaling يحيى داخل الموديل)
            raw_array = self._preprocess_input_data(input_df)
            
            raw_predictions = self.pipeline_session.run(
                [self.output_name], 
                {self.input_name: raw_array}
            )[0]
            
            predictions = raw_predictions.flatten()
            predictions = np.clip(predictions, a_min=0, a_max=None)
            
            logger.info("Full ONNX Pipeline Prediction completed successfully.")
            return predictions

        except Exception as e:
            logger.error("Error in PredictionPipeline.predict method.")
            raise CustomException(e, sys)

    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        """تشغيل التوقع لملف CSV مرجّعاً DataFrame يحتوي على النتائج"""
        try:
            logger.info(f"Reading input file for prediction from: {file_path}")
            df = pd.read_csv(file_path)
            
            preds = self.predict(df)
            
            result_df = df[['unit_number', 'time_in_cycles']].copy()
            result_df['Predicted_RUL'] = np.round(preds, 2)
            
            return result_df

        except Exception as e:
            logger.error("Error in predict_from_file execution.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = PredictionPipeline()
        
        if os.path.exists(test_file_path_cfg):
            results = pipeline.predict_from_file(test_file_path_cfg)
            print("\n>>> Sample Predictions Results (Full ONNX Pipeline) <<<")
            print(results.head(10))
        else:
            print(f"Test file not found at {test_file_path_cfg}. Run training_pipeline first.")
            
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")