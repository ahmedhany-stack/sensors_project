import os
import pytest
import numpy as np
import pandas as pd
import onnxruntime as rt
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

@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_param")
@patch("mlflow.onnx.log_model")
def test_model_trainer_and_prediction(
    mock_log_onnx, mock_log_param, mock_start_run, 
    mock_set_exp, mock_set_uri, sample_training_data, tmp_path, monkeypatch
):
    """اختبار عملية التدريب وتصدير ONNX مع عمل Mock لـ MLflow لضمان السرعة في الـ CI/CD"""
    
    # 1. إعداد مسار حفظ ملف ONNX المؤقت
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgb_rul_model.onnx"

    monkeypatch.setattr("src.components.model_trainer.ModelTrainerConfig.trained_model_file_path", str(model_path))

    # 2. تشغيل التدريب
    trainer = ModelTrainer()
    saved_path = trainer.initiate_model_trainer(sample_training_data)

    # 3. التأكد من إنشاء الملف بنجاح
    assert os.path.exists(saved_path)

    # 4. اختبار التوقع عبر ONNX Runtime
    session = rt.InferenceSession(saved_path)
    input_name = session.get_inputs()[0].name
    
    sample_input = np.array([[518.67, 641.95]], dtype=np.float32)
    predictions = session.run(None, {input_name: sample_input})[0]

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1