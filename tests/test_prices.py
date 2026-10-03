from fastapi.testclient import TestClient
from prices.main import app

def test_known_house():
    payload = TestClient(app).post("/predict", json={"sqft": 1200, "bedrooms": 2}).json()
    assert abs(payload["price"] - 236000) < 1
