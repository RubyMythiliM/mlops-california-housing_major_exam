import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    # Load dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ MSE: {mse:.4f}")

    # Ensure directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Save model
    joblib.dump(model, "artifacts/model.joblib")
    print("✅ Model saved to artifacts/model.joblib")

if __name__ == "__main__":
    main()

