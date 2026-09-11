import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from src.utils.logger import logger
from src.utils.exception import CustomException


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
trans_cfg = config.get("data_transformation", {})
paths_cfg = trans_cfg.get("paths", {})
rul_cfg = trans_cfg.get("rul", {})
fe_cfg = trans_cfg.get("feature_engineering", {})
cols_cfg = trans_cfg.get("columns", {})


@dataclass
class DataTransformationConfig:
    transformed_train_path: str = paths_cfg.get("transformed_train", os.path.join("data", "features", "transformed_train.csv"))
    transformed_test_path: str = paths_cfg.get("transformed_test", os.path.join("data", "features", "transformed_test.csv"))
    scaler_path: str = paths_cfg.get("scaler", os.path.join("models", "scaler.onnx"))
    features_path: str = paths_cfg.get("features_json", os.path.join("models", "features.json"))


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.clip_upper = rul_cfg.get("clip_upper", 125)
        self.rolling_window = fe_cfg.get("rolling_window", 10)
        self.rolling_min_periods = fe_cfg.get("rolling_min_periods", 1)
        self.lag_step = fe_cfg.get("lag_step", 1)
        self.constant_cols = cols_cfg.get("constant_cols", ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19'])
        self.exclude_cols = cols_cfg.get("exclude_cols", ['unit_number', 'time_in_cycles', 'RUL'])

    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب العمر المتبقي للمعدة (Remaining Useful Life - RUL)"""
        max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_in_cycles']
        df['RUL'] = df['RUL'].clip(upper=self.clip_upper)
        return df

    def _create_features(self, df: pd.DataFrame, constant_cols: list) -> pd.DataFrame:
        """استخراج الـ Rolling و Lag Features بعد حذف الأعمدة الثابتة غير المؤثرة"""
        cols_to_drop = [c for c in constant_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        sensor_cols = [c for c in df.columns if c.startswith('s_') or c.startswith('setting_')]
        
        for col in sensor_cols:
            df[f'{col}_roll_mean'] = df.groupby('unit_number')[col].transform(
                lambda x: x.rolling(self.rolling_window, min_periods=self.rolling_min_periods).mean()
            )
            df[f'{col}_roll_std'] = df.groupby('unit_number')[col].transform(
                lambda x: x.rolling(self.rolling_window, min_periods=self.rolling_min_periods).std()
            ).fillna(0)
            
            lag_col_name = f'{col}_lag_{self.lag_step}'
            df[lag_col_name] = df.groupby('unit_number')[col].shift(self.lag_step)
            df[lag_col_name] = df.groupby('unit_number')[lag_col_name].bfill()
            
        return df

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Starting Data Transformation and Feature Engineering process...")
        try:
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Transformation paths are invalid: {train_path} or {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Loaded datasets for transformation. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # 1. حساب الـ RUL للـ Train و Test
            train_df = self._calculate_rul(train_df)
            test_df = self._calculate_rul(test_df)

            # 2. تطبيق الهندسة الزمنية للخصائص (Feature Engineering)
            train_df = self._create_features(train_df, self.constant_cols)
            test_df = self._create_features(test_df, self.constant_cols)

            # 3. تحديد الـ Features الحقيقية للتدريب
            feature_cols = [c for c in train_df.columns if c not in self.exclude_cols]

            # حفظ قائمة الـ features
            os.makedirs(os.path.dirname(self.transformation_config.features_path), exist_ok=True)
            with open(self.transformation_config.features_path, "w") as f:
                json.dump(feature_cols, f, indent=4)
            logger.info(f"Saved {len(feature_cols)} feature names to {self.transformation_config.features_path}")

            # 4. Standard Scaling
            scaler = StandardScaler()
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            
            # 5. تحويل الـ Scaler إلى ONNX وحفظه
            os.makedirs(os.path.dirname(self.transformation_config.scaler_path), exist_ok=True)
            
            initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
            onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)
            
            with open(self.transformation_config.scaler_path, "wb") as f:
                f.write(onnx_scaler.SerializeToString())
                
            logger.info(f"Saved ONNX Scaler successfully to {self.transformation_config.scaler_path}")

            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path), exist_ok=True)
            train_df.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_path, index=False)
            
            logger.info("Data Transformation and Feature Engineering completed successfully.")
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.scaler_path
            )

        except Exception as e:
            logger.error("Error occurred in Data Transformation component.")
            raise CustomException(e, sys)