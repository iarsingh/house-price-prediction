# House Price Prediction

Level: 2 — Data science

Skills: Python, least squares, a prediction endpoint

A two-feature model is fit on a fixed table of square feet and bedrooms. `POST /predict` returns the fitted price. There is no download and no hosted model.

```bash
pip install -r requirements.txt
pytest -q
```
