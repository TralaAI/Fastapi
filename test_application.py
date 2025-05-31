from fastapi.testclient import TestClient
from unittest.mock import patch
from application import app
from unittest.mock import MagicMock

client = TestClient(app)

def test_predict_unauthorized():
    response = client.post("/predict", json={"features": [1, 2, 3]})
    assert response.status_code == 403

@patch("auth.verify_token", return_value=True)
@patch("application.model.predict", return_value=[[42]])

def test_predict_authorized(mock_predict: MagicMock, mock_verify: MagicMock):
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer faketoken"},
        json={"features": [1, 2, 3]}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == [[42]]