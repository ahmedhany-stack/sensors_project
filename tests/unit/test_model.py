import os
import pytest
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
from unittest.mock import patch
from src.components.model_trainer import ModelTrainer

@pytest.fixture
def sample_training_data(tmp_path):
    """إعداد داتا وهمية مصغرة لاختبار عملية تدريب وحفظ وعمل توقعات الموديل"""
    df = pd.DataFrame({
        'unit_number': [1, 1, 2, 2],
        'time_in_cycles': [1, 2, 1, 2],
        's_1': [518.67, 518.67, 518.67, 518.67],
        's_2': [641.82, 642.15, 641.90, 642.00],
        'RUL': [150, 149, 110, 109]
    })
    file_path = tmp_path / "transformed_train.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@patch("mlflow.start_run")
@patch("mlflow.xgboost.log_model")
@patch("mlflow.log_param")
def test_model_trainer_and_prediction(mock_log_param, mock_log_model, mock_start_run, sample_training_data, tmp_path, monkeypatch):
    """اختبار عملية التدريب مع عمل Mock لـ MLflow لتجنب التعليق، وحفظ الموديل والتأكد من سلامة المخرجات"""
    
    # تغيير مسار الحفظ المؤقت عشان التست يكون معزول (Isolated)
    model_dir = tmp_path / "models"
    model_path = model_dir / "xgb_rul_model.joblib"
    
    monkeypatch.setattr("src.components.model_trainer.ModelTrainerConfig.trained_model_file_path", str(model_path))

    trainer = ModelTrainer()
    saved_path = trainer.initiate_model_trainer(sample_training_data)

    # 1. التأكد إن الموديل اتحفظ في المسار الصحيح
    assert os.path.exists(saved_path)

    # 2. تحميل الموديل والتأكد من إنه نوع XGBRegressor
    loaded_model = joblib.load(saved_path)
    assert isinstance(loaded_model, XGBRegressor)

    # 3. اختبار التوقع (Prediction) والتأكد إن المخرجات عبارة عن Array ومنفذة صح
    sample_X = pd.DataFrame({
        's_1': [518.67],
        's_2': [641.95]
    })
    
    predictions = loaded_model.predict(sample_X)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1
    assert isinstance(float(predictions[0]), float)