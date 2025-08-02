import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

def load_data():
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    return r2, mse

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


