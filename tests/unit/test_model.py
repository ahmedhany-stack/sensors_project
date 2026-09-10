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

def test_model_trainer_and_prediction(sample_training_data, tmp_path, monkeypatch):
    """اختبار عملية التدريب، التصدير لـ ONNX، واختبار التوقع بواسطة ONNX Runtime"""
    
    # 1. إعداد مسار حفظ ملف ONNX المؤقت
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgb_rul_model.onnx"

    monkeypatch.setattr("src.components.model_trainer.ModelTrainerConfig.trained_model_file_path", str(model_path))

    # 2. تشغيل عملية التدريب والتصدير
    trainer = ModelTrainer()
    saved_path = trainer.initiate_model_trainer(sample_training_data)

    # 3. التأكد من إنشاء الملف في المسار المخصص
    assert os.path.exists(saved_path)

    # 4. قراءة نموذج ONNX واختبار التوقع (Inference) عبر ONNX Runtime
    session = rt.InferenceSession(saved_path)
    input_name = session.get_inputs()[0].name
    
    # تجهيز المدخلات بنفس عدد الخصائص المتوقعة (s_1, s_2)
    sample_input = np.array([[518.67, 641.95]], dtype=np.float32)
    predictions = session.run(None, {input_name: sample_input})[0]

    # 5. التأكد من صحة مخرجات التوقع
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1
    assert isinstance(float(predictions[0][0]), float)