import pytest
import json
import sys
import os

# Add app directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from app.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ----------------------------
# Test home page
# ----------------------------
def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Heart Disease Prediction" in response.data

# ----------------------------
# Test prediction
# ----------------------------
def test_predict(client):
    sample_input = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 230,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2
    }

    # POST request to /predict
    response = client.post(
        '/predict',
        data=json.dumps(sample_input),
        content_type='application/json'
    )

    # Check response
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["Low risk", "High risk"]
