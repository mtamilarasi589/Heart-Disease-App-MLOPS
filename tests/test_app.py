import pytest
from app.app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Heart Disease Prediction" in response.data

def test_predict(client):
    sample_input = {
        "age": 45, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    }
    response = client.post('/predict', data=json.dumps(sample_input),
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["Low risk", "High risk"]
