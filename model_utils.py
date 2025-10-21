import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error

def generate_synthetic_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "wind_speed": np.random.uniform(0, 20, n),
        "temperature": np.random.uniform(-5, 25, n),
        "humidity": np.random.uniform(30, 100, n),
        "pressure": np.random.uniform(980, 1030, n),
    })
    df["power_output"] = (
        0.5 * df["wind_speed"] ** 3 - 0.1 * df["temperature"] + np.random.normal(0, 5, n)
    )
    return df

def train_and_predict(lat: float, lon: float):
    df = generate_synthetic_data()
    X = df[["wind_speed", "temperature", "humidity", "pressure"]]
    y = df["power_output"]

    base_models = [
        ("rf", RandomForestRegressor(n_estimators=80, random_state=42)),
        ("gbr", GradientBoostingRegressor(random_state=42)),
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )

    scores = cross_val_score(stacking_model, X, y, cv=5, scoring="neg_mean_absolute_error")
    stacking_model.fit(X, y)

    new_data = pd.DataFrame([[10, 15, 60, 1010]], columns=["wind_speed", "temperature", "humidity", "pressure"])
    prediction = stacking_model.predict(new_data)[0]

    return round(prediction, 2)
