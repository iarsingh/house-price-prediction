import csv
import math
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "houses.csv"
FEATURES = ("sqft", "bedrooms")
LIMITS = {"sqft": (200, 10000), "bedrooms": (0, 10)}


class InputError(ValueError):
    pass


def load(path=DATA):
    with open(path, encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def split(rows):
    train = [row for index, row in enumerate(rows) if index % 4 != 3]
    test = [row for index, row in enumerate(rows) if index % 4 == 3]
    return train, test


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


def fit(rows):
    design = [[1.0] + [row[name] for name in FEATURES] for row in rows]
    targets = [row["price"] for row in rows]
    width = len(design[0])
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(width)] for i in range(width)]
    xty = [sum(x[i] * y for x, y in zip(design, targets)) for i in range(width)]
    coefficients = solve(xtx, xty)
    return {"intercept": coefficients[0], **{name: value for name, value in zip(FEATURES, coefficients[1:])}}


def raw_predict(weights, features):
    return weights["intercept"] + sum(weights[name] * features[name] for name in FEATURES)


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


@lru_cache(maxsize=1)
def trained():
    rows = load()
    train, test = split(rows)
    weights = fit(train)
    ranges = {name: (min(row[name] for row in train), max(row[name] for row in train)) for name in FEATURES}
    return {"weights": weights, "test": metrics(weights, test), "train": metrics(weights, train), "ranges": ranges}


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
