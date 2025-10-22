# model_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_models(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, scaler, X_test, y_test

def predict_models(trained_models, scaler, X_new):
    X_scaled = scaler.transform(X_new)
    predictions = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(X_scaled)
    return predictions
