import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger
from src.utils.exception import CustomException

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join("data", "features", "transformed_train.csv")
    transformed_test_path: str = os.path.join("data", "features", "transformed_test.csv")
    scaler_path: str = os.path.join("models", "scaler.joblib")
    features_path: str = os.path.join("models", "features.json")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_in_cycles']
        df['RUL'] = df['RUL'].clip(upper=125)
        return df

    def _create_features(self, df: pd.DataFrame, constant_cols: list) -> pd.DataFrame:
        # حذف الأعمدة الثابتة
        df.drop(columns=[c for c in constant_cols if c in df.columns], inplace=True)
        
        # استخراج أعمدة الحساسات والعدادات
        sensor_cols = [c for c in df.columns if c.startswith('s_') or c.startswith('setting_')]
        
        # حساب الـ Rolling والـ Lag Features بشكل منفصل لكل محرك
        for col in sensor_cols:
            df[f'{col}_roll_mean'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(10, min_periods=1).mean())
            df[f'{col}_roll_std'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
            df[f'{col}_lag_1'] = df.groupby('unit_number')[col].shift(1)
            df[f'{col}_lag_1'] = df.groupby('unit_number')[f'{col}_lag_1'].bfill()
            
        return df

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Starting Data Transformation and Feature Engineering...")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 1. حساب الـ RUL للـ Train و Test
            train_df = self._calculate_rul(train_df)
            test_df = self._calculate_rul(test_df)

            # 2. تطبيق الهندسة الزمنية للخصائص (Feature Engineering)
            constant_cols = ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
            train_df = self._create_features(train_df, constant_cols)
            test_df = self._create_features(test_df, constant_cols)

            # 3. تحديد الـ Features الحقيقية للتدريب (استبعاد IDs والـ Target)
            exclude_cols = ['unit_number', 'time_in_cycles', 'RUL']
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]

            # حفظ قائمة الـ features
            os.makedirs(os.path.dirname(self.transformation_config.features_path), exist_ok=True)
            with open(self.transformation_config.features_path, "w") as f:
                json.dump(feature_cols, f, indent=4)

            # 4. Fit على الـ Train و Transform على الـ Train و Test
            scaler = StandardScaler()
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            
            # 5. حفظ الـ Scaler والبيانات المحولة
            os.makedirs(os.path.dirname(self.transformation_config.scaler_path), exist_ok=True)
            joblib.dump(scaler, self.transformation_config.scaler_path)

            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path), exist_ok=True)
            train_df.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_path, index=False)
            
            logger.info("Data Transformation completed successfully.")
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.scaler_path
            )

        except Exception as e:
            logger.error("Error in Data Transformation.")
            raise CustomException(e, sys)