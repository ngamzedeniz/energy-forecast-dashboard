import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# --- STACKING MODEL OLUŞTURMA ---
def build_stacking_model():
    base_models = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=150, random_state=42))
    ]
    meta_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3)
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    return stacking_model


# --- MODEL EĞİTİMİ ---
def train_stacking_model(df):
    """
    df: pandas DataFrame
        Sütunlar -> ['wind_speed', 'temperature', 'pressure', 'clouds', 'generation_mw']
    """
    X = df[['wind_speed', 'temperature', 'pressure', 'clouds']].values
    y = df['generation_mw'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_stacking_model()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, score


# --- TAHMİN FONKSİYONU ---
def forecast_generation(model, latest_data):
    """
    latest_data: dict
        {'wind_speed': float, 'temperature': float, 'pressure': float, 'clouds': float}
    """
    X_pred = np.array([[latest_data['wind_speed'], latest_data['temperature'],
                        latest_data['pressure'], latest_data['clouds']]])
    y_pred = model.predict(X_pred)[0]
    return round(float(y_pred), 2)
