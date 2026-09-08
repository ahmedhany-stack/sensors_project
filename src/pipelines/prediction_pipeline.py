import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class PredictionPipelineConfig:
    # تغيير امتداد النموذج إلى .onnx
    model_path: str = os.path.join("models", "xgb_rul_model.onnx")
    scaler_path: str = os.path.join("models", "scaler.joblib")
    features_path: str = os.path.join("models", "features.json")

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self._load_artifacts()

    def _load_artifacts(self):
        """تحميل الـ ONNX Session والـ Scaler وقائمة الـ Features المسجلة أثناء التدريب"""
        try:
            logger.info("Loading inference artifacts (ONNX Model Session, Scaler, Features list)...")
            
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"ONNX Model file not found at {self.config.model_path}")
            if not os.path.exists(self.config.scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {self.config.scaler_path}")
            if not os.path.exists(self.config.features_path):
                raise FileNotFoundError(f"Features JSON file not found at {self.config.features_path}")

            # -------------------------------------------------------------
            # 1. إنشاء ONNX Runtime Session بدلاً من joblib.load للموديل
            # -------------------------------------------------------------
            self.session = ort.InferenceSession(
                self.config.model_path, 
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # 2. تحميل الـ Scaler والـ Features كالمعتاد
            self.scaler = joblib.load(self.config.scaler_path)
            
            with open(self.config.features_path, "r") as f:
                self.feature_names = json.load(f)

            logger.info("Artifacts loaded successfully.")
        except Exception as e:
            logger.error("Failed to load artifacts in PredictionPipeline.")
            raise CustomException(e, sys)

    def _preprocess_input_data(self, df: pd.DataFrame) -> np.ndarray:
        """تطبيق نفس خطوات Feature Engineering والـ Scaler تماماً وتحويلها لـ float32"""
        try:
            df = df.copy()

            # 1. نفس قائمة الأعمدة الثابتة التي تم حذفها أثناء التدريب
            constant_cols = ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
            df.drop(columns=[c for c in constant_cols if c in df.columns], inplace=True)

            # 2. استخراج الحساسات وتطبيق الـ Rolling والـ Lag تماماً مثل DataTransformation
            sensor_cols = [c for c in df.columns if c.startswith('s_') or c.startswith('setting_')]
            
            for col in sensor_cols:
                df[f'{col}_roll_mean'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(10, min_periods=1).mean())
                df[f'{col}_roll_std'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
                df[f'{col}_lag_1'] = df.groupby('unit_number')[col].shift(1)
                df[f'{col}_lag_1'] = df.groupby('unit_number')[f'{col}_lag_1'].bfill()

            # 3. التأكد من تطابق الأعمدة مع الـ features.json المسجل
            missing_cols = set(self.feature_names) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required feature columns in input data: {missing_cols}")

            # ترتيب الأعمدة بنفس الترتيب تماماً وقت التدريب
            X_df = df[self.feature_names].copy()

            # 4. تطبيق الـ Scaler المحفوظ فقط (Transform بدون Fit)
            X_scaled = self.scaler.transform(X_df)

            # -------------------------------------------------------------
            # تحويل البيانات بشكل صريح لـ float32 للإنتاج مع ONNX
            # -------------------------------------------------------------
            return X_scaled.astype(np.float32)

        except Exception as e:
            logger.error("Error during input data preprocessing in PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """تشغيل التوقع لبيانات مدخلة كـ DataFrame باستخدام ONNX Runtime"""
        try:
            logger.info(f"Starting ONNX prediction for input data of shape {input_df.shape}...")
            
            # معالجة البيانات وتحويلها لـ float32 Numpy Array
            processed_array = self._preprocess_input_data(input_df)
            
            # -------------------------------------------------------------
            # إجراء التوقع عبر ONNX Runtime Session
            # -------------------------------------------------------------
            raw_predictions = self.session.run(
                [self.output_name], 
                {self.input_name: processed_array}
            )[0]
            
            predictions = raw_predictions.flatten()
            
            # ضمان أن الـ RUL لا يقل عن 0
            predictions = np.clip(predictions, a_min=0, a_max=None)
            
            logger.info("ONNX Prediction completed successfully.")
            return predictions

        except Exception as e:
            logger.error("Error in PredictionPipeline.predict method.")
            raise CustomException(e, sys)

    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        """تشغيل التوقع لملف CSV مرجّعاً DataFrame يحتوي على النتائج مرتبطة بكل محرك و Session"""
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
    # كود سريع لتجربة الـ Inference Pipeline على داتا الـ Test
    try:
        pipeline = PredictionPipeline()
        test_file = os.path.join("data", "processed", "test.csv")
        
        if os.path.exists(test_file):
            results = pipeline.predict_from_file(test_file)
            print("\n>>> Sample Predictions Results (ONNX) <<<")
            print(results.head(10))
        else:
            print(f"Test file not found at {test_file}. Run training_pipeline first.")
            
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")