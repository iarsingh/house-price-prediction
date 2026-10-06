# house-price-prediction — interview questions and answers

[README](README.md) · [Project architecture](PROJECT_ARCHITECTURE.md)

Answers below use this repository’s files and implementation. They distinguish existing behavior from suggested extensions; source links let you verify each walkthrough.

## 1. What problem does house-price-prediction address, and what can you demonstrate?

`data/houses.csv` has 16 sales with square feet, bedrooms, and price. Every fourth row is held out for testing. The model is ordinary least squares with an intercept, solved from the normal equations with Gaussian elimination. There is no NumPy and no hosted model, so every number can be checked by hand.

I would demonstrate the linked implementation or examples and distinguish that evidence from any planned production features. Start with [`README.md`](README.md).

## 2. How is this repository organized?

- [`src/prices/main.py`](src/prices/main.py): Implementation or supporting configuration.
- [`src/prices/ops.py`](src/prices/ops.py): Implementation or supporting configuration.
- [`src/prices/model.py`](src/prices/model.py): Implementation or supporting configuration.
- [`requirements.txt`](requirements.txt): Implementation or supporting configuration.
- [`src/prices/__init__.py`](src/prices/__init__.py): Implementation or supporting configuration.
- [`Dockerfile`](Dockerfile): Container build/service configuration.
- [`Makefile`](Makefile): Implementation or supporting configuration.
- [`docker-compose.yml`](docker-compose.yml): Container build/service configuration.

[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) contains the component diagram and the implementation walkthrough.

## 3. Can you walk through `predict` and explain the decision it makes?

The main walkthrough here is `predict(features)` in [`src/prices/model.py`](src/prices/model.py#L79).

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

The implementation calls `InputError`, `isinstance`, `model['weights'].items`, `raw_predict`, `round`, `trained`. In an interview, trace those calls in execution order using a fixture input.

## 4. What responsibility does `solve` have?

`solve(matrix, vector)` is defined in [`src/prices/model.py`](src/prices/model.py#L26).

Its return expressions include:

- `[grid[row][size] / grid[row][row] for row in range(size)]`

It uses `InputError`, `abs`, `len`, `max`, `range`, `zip`. This is the code path I would compare against the caller to explain responsibility boundaries.

## 5. What input validation and failure behavior are implemented?

Explicit failure paths include:

- `HTTPException(status_code=422, detail='houses must be a list of 1 to 500 items')` in [`src/prices/main.py`](src/prices/main.py#L38).
- `HTTPException(status_code=422, detail=str(exc))` in [`src/prices/main.py`](src/prices/main.py#L31).
- `HTTPException(status_code=422, detail=str(exc))` in [`src/prices/main.py`](src/prices/main.py#L42).
- `InputError('features are collinear; the model cannot be fit')` in [`src/prices/model.py`](src/prices/model.py#L32).
- `InputError(f'{name} is required')` in [`src/prices/model.py`](src/prices/model.py#L82).
- `InputError(f'{name} must be a number from {low} to {high}')` in [`src/prices/model.py`](src/prices/model.py#L86).
- `HTTPException(status_code=404, detail='workspace not found')` in [`src/prices/ops.py`](src/prices/ops.py#L77).

I would test both the condition that reaches each exception and the caller that translates it. An explicit raise does not mean every malformed input or dependency failure is handled.

## 6. Which test would you use to demonstrate correctness?

[`tests/test_ops.py`](tests/test_ops.py#L8) contains `test_readyz`:

```python
def test_readyz():
    r = client.get("/v1/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
```

This is a concrete regression example from the repository. Its assertions establish that case; they do not establish behavior for every input or under production load.

## 7. What HTTP interface does the code expose?

- `GET /healthz` → `healthz` in [`src/prices/main.py`](src/prices/main.py#L11).
- `GET /model` → `model` in [`src/prices/main.py`](src/prices/main.py#L16).
- `POST /predict` → `post_predict` in [`src/prices/main.py`](src/prices/main.py#L27).
- `POST /predict/batch` → `post_batch` in [`src/prices/main.py`](src/prices/main.py#L35).
- `GET /readyz` → `readyz` in [`src/prices/ops.py`](src/prices/ops.py#L74).
- `POST /workspaces` → `create_workspace` in [`src/prices/ops.py`](src/prices/ops.py#L80).
- `GET /workspaces` → `list_workspaces` in [`src/prices/ops.py`](src/prices/ops.py#L98).
- `POST /workspaces/{workspace_id}/jobs` → `create_job` in [`src/prices/ops.py`](src/prices/ops.py#L106).

These are literal decorators. Application/router prefixes, authentication, and middleware must be checked in the corresponding setup code.

## 8. Where does state live, and what happens with multiple workers?

Module-level containers include `LIMITS` in [`src/prices/model.py`](src/prices/model.py); `_WORKSPACES`, `_JOBS`, `_AUDIT`, `_METRICS` in [`src/prices/ops.py`](src/prices/ops.py).

These containers belong to a Python process. Inspect which are constant fixtures and which are mutated. Mutable process state needs an explicit shared-storage or synchronization strategy before multiple workers can provide consistent behavior.

## 9. How would another engineer reproduce your walkthrough?

Start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

These commands follow repository manifests; environment setup and command results still need to be checked on the target machine.

## 10. What does automation verify, and what does it not prove?

Inspect [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for triggers, permissions, and job commands. I would name the checks that those definitions run and show the latest run separately. A workflow definition alone does not establish a successful deployment, security review, or production SLO.

## 11. How would you present this project in a Forward Deployed Engineer interview?

Start with the user and operational problem described in [`README.md`](README.md). Explain one constraint that changes the implementation, show the linked code or example, and walk through a success case and a failure case. Agree on a measurable acceptance criterion before expanding the solution, and leave a handoff with data boundaries and rollback ownership. Any proposed production or business metric should be identified as a target until measured.

## 12. What is the input-to-output contract of `predict`?

In [`src/prices/model.py`](src/prices/model.py#L79), `predict(features)` receives the inputs. The function computes these intermediate values:

- `model = trained()`
- `outside = [name for name in FEATURES if not model['ranges'][name][0] <= features[name] <= model['ranges'][name][1]]`

Its result is defined by:

- `{'price': round(raw_predict(model['weights'], features), 2), 'extrapolated': outside, 'weights': {key: round(value, 4) for key, value in model['weights'].items()}}`

## 13. Which decision rules or boundary conditions should an interviewer challenge?

The implementation in [`src/prices/model.py`](src/prices/model.py#L79) branches on:

- `name not in features`
- `not isinstance(value, (int, float)) or isinstance(value, bool) or (not low <= value <= high)`

A useful extension is a table-driven test that covers each condition just below, at, and above its boundary where applicable. These expressions are the current rules; changing them changes behavior and should be justified by the project’s acceptance criteria.

## 14. What does the operations plane add, and where is its limit?

[`src/prices/ops.py`](src/prices/ops.py) declares `GET /readyz`, `POST /workspaces`, `GET /workspaces`, `POST /workspaces/{workspace_id}/jobs`, `GET /jobs/{job_id}`, `POST /jobs/{job_id}/approve`, `GET /audit`, `GET /metrics`. Inspect the application’s `include_router` call for its URL prefix.

Its state containers are `_WORKSPACES`, `_JOBS`, `_AUDIT`, `_METRICS`. The job-approval handler defines whether a target is accepted or refused; check that branch and the associated tests instead of treating a recorded job as a successful infrastructure apply.
