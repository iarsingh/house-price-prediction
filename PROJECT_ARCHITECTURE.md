# house-price-prediction — project architecture

[README](README.md) · [Interview questions and answers](INTERVIEW_QA.md)

## Purpose and scope

`data/houses.csv` has 16 sales with square feet, bedrooms, and price. Every fourth row is held out for testing. The model is ordinary least squares with an intercept, solved from the normal equations with Gaussian elimination. There is no NumPy and no hosted model, so every number can be checked by hand.

This document describes files and symbols in this checkout. Deployment templates and statements in the original overview are distinguished from a verified running environment.

## Component diagram

```mermaid
flowchart LR
    M0["src/prices/__init__.py"]
    M1["src/prices/main.py"]
    M2["src/prices/model.py"]
    M1 -->|imports| M2
```

For Python repositories, arrows show resolved local imports, not network calls or deployment order. Otherwise the diagram is a repository component map; containment arrows do not assert runtime integration.

## Components and responsibilities

| Component | Responsibility |
| --- | --- |
| [`src/prices/main.py`](src/prices/main.py) | HTTP handlers: `GET /healthz`, `GET /model`, `POST /predict`, `POST /predict/batch` |
| [`src/prices/model.py`](src/prices/model.py) | Functions: `load`, `split`, `solve`, `fit`, `raw_predict`, `metrics`, `trained` |
| [`requirements.txt`](requirements.txt) | Implementation or supporting configuration |
| [`src/prices/__init__.py`](src/prices/__init__.py) | Implementation or supporting configuration |
| [`tests/test_prices.py`](tests/test_prices.py) | Executable checks and regression examples |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions job definitions |
| [`README.md`](README.md) | Project explanations or operating notes |

## Request interface

| Method and path | Handler | Source |
| --- | --- | --- |
| `GET /healthz` | `healthz` | [`src/prices/main.py`](src/prices/main.py#L9) |
| `GET /model` | `model` | [`src/prices/main.py`](src/prices/main.py#L14) |
| `POST /predict` | `post_predict` | [`src/prices/main.py`](src/prices/main.py#L25) |
| `POST /predict/batch` | `post_batch` | [`src/prices/main.py`](src/prices/main.py#L33) |

The table lists literal route decorators found in the inspected Python modules. Router prefixes and middleware can add behavior; check the linked handler and application setup before calling an endpoint.

## Implementation walkthrough

### `predict(features)`

Source: [`src/prices/model.py`](src/prices/model.py#L79).

Calls visible in this function: `InputError`, `isinstance`, `model['weights'].items`, `raw_predict`, `round`, `trained`.

```python
def predict(features):
    for name in FEATURES:
        if name not in features:
            raise InputError(f"{name} is required")
        low, high = LIMITS[name]
        value = features[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
            raise InputError(f"{name} must be a number from {low} to {high}")
    model = trained()
    outside = [name for name in FEATURES if not model["ranges"][name][0] <= features[name] <= model["ranges"][name][1]]
    return {
        "price": round(raw_predict(model["weights"], features), 2),
        "extrapolated": outside,
        "weights": {key: round(value, 4) for key, value in model["weights"].items()},
    }
```

### `solve(matrix, vector)`

Source: [`src/prices/model.py`](src/prices/model.py#L26).

Calls visible in this function: `InputError`, `abs`, `len`, `max`, `range`, `zip`.

```python
def solve(matrix, vector):
    size = len(vector)
    grid = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(grid[row][column]))
        if abs(grid[pivot][column]) < 1e-12:
            raise InputError("features are collinear; the model cannot be fit")
        grid[column], grid[pivot] = grid[pivot], grid[column]
        for row in range(size):
            if row != column:
                factor = grid[row][column] / grid[column][column]
                grid[row] = [a - factor * b for a, b in zip(grid[row], grid[column])]
    return [grid[row][size] / grid[row][row] for row in range(size)]
```

### `metrics(weights, rows)`

Source: [`src/prices/model.py`](src/prices/model.py#L55).

Calls visible in this function: `abs`, `len`, `math.sqrt`, `raw_predict`, `round`, `sum`, `zip`.

```python
def metrics(weights, rows):
    actual = [row["price"] for row in rows]
    predicted = [raw_predict(weights, row) for row in rows]
    errors = [a - p for a, p in zip(actual, predicted)]
    mean = sum(actual) / len(actual)
    total = sum((a - mean) ** 2 for a in actual)
    residual = sum(error ** 2 for error in errors)
    return {
        "mae": round(sum(abs(error) for error in errors) / len(errors), 2),
        "rmse": round(math.sqrt(residual / len(errors)), 2),
        "r2": round(1 - residual / total, 4) if total else None,
        "rows": len(rows),
    }
```

### `fit(rows)`

Source: [`src/prices/model.py`](src/prices/model.py#L41).

Calls visible in this function: `len`, `range`, `solve`, `sum`, `zip`.

```python
def fit(rows):
    design = [[1.0] + [row[name] for name in FEATURES] for row in rows]
    targets = [row["price"] for row in rows]
    width = len(design[0])
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(width)] for i in range(width)]
    xty = [sum(x[i] * y for x, y in zip(design, targets)) for i in range(width)]
    coefficients = solve(xtx, xty)
    return {"intercept": coefficients[0], **{name: value for name, value in zip(FEATURES, coefficients[1:])}}
```

## Validation and failure paths

| Explicit exception | Source |
| --- | --- |
| `HTTPException(status_code=422, detail='houses must be a list of 1 to 500 items')` | [`src/prices/main.py`](src/prices/main.py#L36) |
| `HTTPException(status_code=422, detail=str(exc))` | [`src/prices/main.py`](src/prices/main.py#L29) |
| `HTTPException(status_code=422, detail=str(exc))` | [`src/prices/main.py`](src/prices/main.py#L40) |
| `InputError('features are collinear; the model cannot be fit')` | [`src/prices/model.py`](src/prices/model.py#L32) |
| `InputError(f'{name} is required')` | [`src/prices/model.py`](src/prices/model.py#L82) |
| `InputError(f'{name} must be a number from {low} to {high}')` | [`src/prices/model.py`](src/prices/model.py#L86) |

These are explicit exceptions in the inspected source, rather than a claim that every failure is handled. Follow the calling handler to see whether the exception becomes an HTTP response or propagates.

## Data and state

- [`src/prices/model.py`](src/prices/model.py) defines module-level containers: `LIMITS`.

Module-level dictionaries/lists live in a Python process. They can be fixtures or mutable state; inspect writes before treating them as persistent storage. A production extension would need to define persistence and concurrency behavior explicitly.

## Data flow and design decisions

### What is the input-to-output contract of `predict`

In [`src/prices/model.py`](src/prices/model.py#L79), `predict(features)` receives the inputs. The function computes these intermediate values:

- `model = trained()`
- `outside = [name for name in FEATURES if not model['ranges'][name][0] <= features[name] <= model['ranges'][name][1]]`

Its result is defined by:

- `{'price': round(raw_predict(model['weights'], features), 2), 'extrapolated': outside, 'weights': {key: round(value, 4) for key, value in model['weights'].items()}}`

### Which decision rules or boundary conditions should an interviewer challenge

The implementation in [`src/prices/model.py`](src/prices/model.py#L79) branches on:

- `name not in features`
- `not isinstance(value, (int, float)) or isinstance(value, bool) or (not low <= value <= high)`

A useful extension is a table-driven test that covers each condition just below, at, and above its boundary where applicable. These expressions are the current rules; changing them changes behavior and should be justified by the project’s acceptance criteria.

## Setup and verification

The following commands are derived from the checked-in dependency/test contracts. Execute them from the repository root; the block prepares a local environment, not a cloud deployment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Python dependencies: [`requirements.txt`](requirements.txt).

Test entry points: [`tests/test_prices.py`](tests/test_prices.py).

Automation definitions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Read their triggers and job steps to determine what CI actually runs.

## Operating boundaries and design review

Before turning this checkout into a customer deployment, establish the input contract, data ownership, access controls, failure response, evaluation criteria, and rollback owner. Repository fixtures and unit tests demonstrate local behavior; they do not establish throughput, uptime, compliance, or business impact.

A useful architecture review starts with the linked implementation: identify where input enters, where a decision is made, which state can change, and which external dependency can fail. Add a deployment view only for infrastructure that is actually configured and exercised.
