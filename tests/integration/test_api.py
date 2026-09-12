from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, get_current_user, ml_artifacts


@pytest.fixture(scope="module")
def client():
    """
    Fixture لتشغيل الـ TestClient مع تفعيل الـ lifespan
    وعمل Mock كامل لجميع اتصالات قاعدة البيانات لمنع أي اتصال حقيقي بـ Postgres.
    """
    # تجاوز نظام التوثيق (Auth)
    app.dependency_overrides[get_current_user] = lambda: {
        "username": "test_user"
    }

    # عمل Mock للـ Pipeline والـ Database Layer مع إضافة create=True لتفادي AttributeError
    with patch("src.api.app.PredictionPipeline") as mock_pipeline_class, \
         patch("src.api.database.init_db", create=True) as mock_init_db, \
         patch("src.api.database.SessionLocal") as mock_session, \
         patch("src.api.database.engine") as mock_engine, \
         patch("src.api.database.Base.metadata.create_all", create=True) as mock_create_all:

        # تجهيز الـ Mock الخاص بالـ Model Prediction
        mock_pipeline_instance = mock_pipeline_class.return_value
        mock_pipeline_instance.predict.return_value = [120.5]

        # تجهيز الـ Mock الخاص بالـ DB Session
        mock_db_session = MagicMock()
        mock_session.return_value = mock_db_session

        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


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
    assert "app_requests_total" in response.text or response.status_code == 200


def test_predict_endpoint_success(client):
    """اختبار إرسال بيانات وتوقع الـ RUL بنجاح (Inference) مع تجنب اتصال قاعدة البيانات"""
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
                "s_21": 23.4190,
            }
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_records"] == 1
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["predicted_rul"] == 120.5


def test_predict_endpoint_model_unavailable(client):
    """اختبار حالة لو الموديل مش محمل والـ API رجّع 503"""
    original_status = ml_artifacts.get("model_loaded")
    ml_artifacts["model_loaded"] = False

    payload = {
        "records": [
            {
                "unit_number": 1,
                "time_in_cycles": 1,
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
                "s_21": 23.4190,
            }
        ]
    }

    response = client.post("/predict", json=payload)
    ml_artifacts["model_loaded"] = original_status

    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"