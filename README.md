# House Price Prediction

<!-- project-guide:start -->
## Project guide

[Project architecture](PROJECT_ARCHITECTURE.md) · [Interview questions and answers](INTERVIEW_QA.md)

Use the architecture document for the component diagram, implementation boundaries, and verification entry points. The interview guide includes source-backed answers and project walkthroughs.

### Implementation map

| Component | Responsibility |
| --- | --- |
| [`src/prices/main.py`](src/prices/main.py) | HTTP handlers: `GET /healthz`, `GET /model`, `POST /predict`, `POST /predict/batch` |
| [`src/prices/model.py`](src/prices/model.py) | Functions: `load`, `split`, `solve`, `fit`, `raw_predict`, `metrics`, `trained` |
| [`requirements.txt`](requirements.txt) | Implementation or supporting configuration |
| [`src/prices/__init__.py`](src/prices/__init__.py) | Implementation or supporting configuration |
| [`tests/test_prices.py`](tests/test_prices.py) | Executable checks and regression examples |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions job definitions |
| [`README.md`](README.md) | Project explanations or operating notes |

### Local setup and verification

From the repository root (the commands follow the checked-in manifests):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

To serve the FastAPI application locally, install the server separately if it is not already available:

```bash
python -m pip install uvicorn
PYTHONPATH=src python -m uvicorn prices.main:app --reload
```

<!-- project-guide:end -->

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
