from fastapi.testclient import TestClient
from src.api import app, CONSERVATIVE_THRESHOLD

client = TestClient(app)


def test_health_endpoint():
    """Verifica que /health responde 200 OK y confirma la carga del modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["threshold"] == CONSERVATIVE_THRESHOLD


def test_predict_valid_payload_and_threshold_logic():
    """
    Comprueba el contrato de salida de /predict y verifica que la decisión binaria
    respeta estrictamente el umbral conservador (0.65).
    """
    payload = {
        "Platform": "PS4",
        "Year": 2018,
        "Genre": "Action",
        "Publisher": "Electronic Arts",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"is_hit", "probability", "threshold", "recommendation"}
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold"] == CONSERVATIVE_THRESHOLD

    expected_hit = int(data["probability"] >= CONSERVATIVE_THRESHOLD)
    assert data["is_hit"] == expected_hit


def test_predict_invalid_year_returns_422():
    """Verifica que años fuera del rango [1980, 2026] son rechazados por Pydantic."""
    payload = {
        "Platform": "PC",
        "Year": 1950,
        "Genre": "Strategy",
        "Publisher": "Indie_Studio",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_strings_returns_422():
    """Verifica que campos de texto vacíos o con solo espacios son rechazados."""
    payload = {
        "Platform": "   ",
        "Year": 2025,
        "Genre": "Action",
        "Publisher": "Ubisoft",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422