# House Price Prediction

Level: 2 — Data science

Skills: Python, linear regression from scratch, train and test split, MAE, RMSE, R²

`data/houses.csv` has 16 sales with square feet, bedrooms, and price. Every fourth row is held out for testing. The model is ordinary least squares with an intercept, solved from the normal equations with Gaussian elimination. There is no NumPy and no hosted model, so every number can be checked by hand.

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src uvicorn prices.main:app --reload
```

| Method and path | Returns |
| --- | --- |
| `GET /model` | Weights, train and test metrics, and the training range of each feature |
| `POST /predict` | `sqft` and `bedrooms` in, a price out, plus any feature outside the training range |
| `POST /predict/batch` | Up to 500 houses at once |

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"sqft":1200,"bedrooms":2}'
```

## What it refuses or flags

- A missing feature, a value that is not a number, square feet outside 200 to 10,000, or bedrooms outside 0 to 10.
- A house outside the training range is still priced, but `extrapolated` names the feature, so nobody mistakes a guess for an interpolation.
- Collinear features that make the normal equations unsolvable.
