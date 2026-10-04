from fastapi import FastAPI, HTTPException

from prices.model import InputError, predict, trained

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/model")
def model():
    body = trained()
    return {
        "weights": {key: round(value, 4) for key, value in body["weights"].items()},
        "train": body["train"],
        "test": body["test"],
        "ranges": body["ranges"],
    }


@app.post("/predict")
def post_predict(body: dict):
    try:
        return predict(body)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch")
def post_batch(body: dict):
    houses = body.get("houses")
    if not isinstance(houses, list) or not 1 <= len(houses) <= 500:
        raise HTTPException(status_code=422, detail="houses must be a list of 1 to 500 items")
    try:
        return {"predictions": [predict(house) for house in houses]}
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
