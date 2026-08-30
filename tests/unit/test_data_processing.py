import os
import pytest
import pandas as pd
import numpy as np
import joblib
import json
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

@pytest.fixture
def sample_raw_data(tmp_path):
    """إعداد عينة داتا خام وهمية تحاكي داتا السنسورات (CMAPSS format)"""
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 2, 2, 2],
        'time_in_cycles': [1, 2, 3, 1, 2, 3],
        'setting_1': [0.0007, 0.0001, -0.0003, 0.0010, 0.0005, -0.0002],
        'setting_2': [0.0000, 0.0002, -0.0001, 0.0001, -0.0003, 0.0004],
        'setting_3': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        's_1': [518.67, 518.67, 518.67, 518.67, 518.67, 518.67],
        's_2': [641.82, 642.15, 642.30, 641.90, 642.00, 642.10],
        's_5': [14.62, 14.62, 14.62, 14.62, 14.62, 14.62],
        's_10': [1.3, 1.3, 1.3, 1.3, 1.3, 1.3],
        's_16': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        's_18': [2388.0, 2388.0, 2388.0, 2388.0, 2388.0, 2388.0],
        's_19': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    })
    
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    df.to_csv(train_file, index=False)
    df.to_csv(test_file, index=False)
    
    return str(train_file), str(test_file)

def test_data_validation(sample_raw_data):
    """اختبار مكون التحقق من صحة البيانات (DataValidation) وخلوها من الـ Missing values"""
    train_path, _ = sample_raw_data
    validator = DataValidation()
    
    status = validator.validate_all_columns(train_path)
    assert status is True
    assert os.path.exists(validator.validation_config.status_file_path)

def test_data_transformation_pipeline(sample_raw_data):
    """اختبار كامل لعملية تحويل البيانات، حساب الـ RUL، هندسة الخصائص، وحفظ الـ Scaler"""
    train_path, test_path = sample_raw_data
    
    transformer = DataTransformation()
    res_train, res_test, res_scaler = transformer.initiate_data_transformation(train_path, test_path)
    
    # 1. التأكد من إنشاء الملفات الناتجة الأساسية
    assert os.path.exists(res_train)
    assert os.path.exists(res_test)
    assert os.path.exists(res_scaler)
    
    # 2. قراءة الداتا المحولة والتأكد من وجود عمود الـ RUL وحساباته
    transformed_df = pd.read_csv(res_train)
    assert 'RUL' in transformed_df.columns
    assert transformed_df.loc[0, 'RUL'] == 2
    
    # 3. التأكد من حذف الأعمدة الثابتة المحددة مسبقاً
    assert 's_1' not in transformed_df.columns
    assert 'setting_3' not in transformed_df.columns
    
    # 4. التأكد من عمل الـ Rolling والـ Lags Features
    assert any(col.endswith('_roll_mean') for col in transformed_df.columns)
    assert any(col.endswith('_lag_1') for col in transformed_df.columns)
    
    # 5. التأكد من عمل الـ Scaler بنجاح
    scaler = joblib.load(res_scaler)
    assert scaler is not None