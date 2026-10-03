TRAINING = (
    (1000, 2, 200000),
    (1500, 3, 300000),
    (800, 1, 154000),
    (2000, 4, 400000),
)

def fit(rows=TRAINING):
    s11 = s12 = s22 = t1 = t2 = 0.0
    for sqft, bedrooms, price in rows:
        s11 += sqft * sqft
        s12 += sqft * bedrooms
        s22 += bedrooms * bedrooms
        t1 += sqft * price
        t2 += bedrooms * price
    det = s11 * s22 - s12 * s12
    sqft_weight = (t1 * s22 - t2 * s12) / det
    bedroom_weight = (s11 * t2 - s12 * t1) / det
    return {"sqft": sqft_weight, "bedrooms": bedroom_weight}

def predict(features):
    weights = fit()
    price = weights["sqft"] * features["sqft"] + weights["bedrooms"] * features["bedrooms"]
    return {"price": round(price, 2), "weights": {key: round(value, 4) for key, value in weights.items()}}
