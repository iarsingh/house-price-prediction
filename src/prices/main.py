from fastapi import FastAPI
from prices.model import predict

app = FastAPI()

@app.post("/predict")
def post_predict(body: dict):
    return predict(body)
