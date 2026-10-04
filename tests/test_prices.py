import pytest
from fastapi.testclient import TestClient

from prices.main import app
from prices.model import InputError, fit, solve, split, load

client = TestClient(app)


def test_known_house_is_close_to_the_generating_formula():
    payload = client.post("/predict", json={"sqft": 1200, "bedrooms": 2}).json()
    assert abs(payload["price"] - 214000) / 214000 < 0.05
    assert payload["extrapolated"] == []


def test_held_out_rows_score_well():
    model = client.get("/model").json()
    assert model["test"]["rows"] == 4
    assert model["train"]["rows"] == 12
    assert model["test"]["r2"] > 0.95
    assert model["test"]["mae"] < 10000


def test_fit_recovers_an_exact_line():
    rows = [{"sqft": s, "bedrooms": b, "price": 1000 + 100 * s + 5000 * b} for s, b in [(500, 1), (900, 2), (1300, 2), (2000, 4)]]
    weights = fit(rows)
    assert weights["intercept"] == pytest.approx(1000)
    assert weights["sqft"] == pytest.approx(100)
    assert weights["bedrooms"] == pytest.approx(5000)


def test_collinear_features_are_refused():
    with pytest.raises(InputError, match="collinear"):
        solve([[1, 2], [2, 4]], [1, 2])


def test_split_is_deterministic():
    train, test = split(load())
    assert [row["sqft"] for row in test] == [1250, 1850, 1300, 2400]
    assert len(train) == 12


def test_outside_training_range_is_flagged():
    payload = client.post("/predict", json={"sqft": 5000, "bedrooms": 2}).json()
    assert payload["extrapolated"] == ["sqft"]


def test_missing_and_impossible_inputs_are_refused():
    assert client.post("/predict", json={"sqft": 1200}).status_code == 422
    assert client.post("/predict", json={"sqft": 50, "bedrooms": 2}).status_code == 422
    assert client.post("/predict", json={"sqft": "big", "bedrooms": 2}).status_code == 422


def test_batch_predictions():
    houses = [{"sqft": 1000, "bedrooms": 2}, {"sqft": 2000, "bedrooms": 4}]
    predictions = client.post("/predict/batch", json={"houses": houses}).json()["predictions"]
    assert predictions[0]["price"] < predictions[1]["price"]
    assert client.post("/predict/batch", json={"houses": []}).status_code == 422
