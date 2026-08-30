import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.app import app, ml_artifacts

@pytest.fixture(scope="module")
def client():
    """Fixture لتشغيل الـ TestClient مع تفعيل الـ lifespan مسبقاً وتعديل مسار الـ patch"""
    with patch("src.api.app.PredictionPipeline") as mock_pipeline_class:
        mock_pipeline_instance = mock_pipeline_class.return_value
        mock_pipeline_instance.predict.return_value = [120.5, 85.2]
        
        with TestClient(app) as c:
            yield c

def test_root_endpoint(client):
    """اختبار نقطة البداية والتأكد من رجوع الـ Documentation والـ Metrics"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["swagger_docs"] == "/docs"
    assert data["metrics"] == "/metrics"

def test_health_check_endpoint(client):
    """اختبار الـ Health Check والتأكد من حالة الموديل"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["version"] == "1.0.0"

def test_metrics_endpoint(client):
    """اختبار نقطة تجميع الـ Prometheus Metrics"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text

def test_predict_endpoint_success(client):
    """اختبار إرسال بيانات وتوقع الـ RUL بنجاح (Inference) مع استخدام أسماء الحقول الصحيحة s_1, s_2..."""
    payload = {
        "records": [
            {
                "unit_number": 1,
                "time_in_cycles": 10,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
                "s_1": 518.67,
                "s_2": 641.82,
                "s_3": 1589.70,
                "s_4": 1400.60,
                "s_5": 14.62,
                "s_6": 21.61,
                "s_7": 554.36,
                "s_8": 2388.06,
                "s_9": 9046.19,
                "s_10": 1.30,
                "s_11": 47.47,
                "s_12": 521.66,
                "s_13": 2388.02,
                "s_14": 8138.62,
                "s_15": 8.4195,
                "s_16": 0.03,
                "s_17": 392.0,
                "s_18": 2388.0,
                "s_19": 100.0,
                "s_20": 39.06,
                "s_21": 23.4190
            }
        ]
    }
    
    # عمل Mock لدالة حفظ الداتا في الدेटाبيس عشان التيست يكون مستقل
    with patch("src.api.app.save_predictions_to_db") as mock_save_db:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_records"] == 1
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["predicted_rul"] == 120.5
        
        mock_save_db.assert_called_once()

def test_predict_endpoint_model_unavailable(client):
    """اختبار حالة لو الموديل مش محمل والـ API ضرب 503"""
    original_status = ml_artifacts.get("model_loaded")
    ml_artifacts["model_loaded"] = False
    
    # استخدام Payload صحيح كحقول عشان الـ validation يعدي ويظهر الـ 503 الخاصة بالموديل
    payload = {
        "records": [
            {
                "unit_number": 1,
                "time_in_cycles": 1,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
                "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
                "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0, "s_20": 39.06,
                "s_21": 23.4190
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    ml_artifacts["model_loaded"] = original_status
    
    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"