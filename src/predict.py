import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

def load_model(path="model.joblib"):
    return joblib.load(path)

def main():
    model = load_model()

    data = fetch_california_housing()
    X = data.data
    y = data.target

    y_pred = model.predict(X)
    print("✅ Model loaded successfully.")
    print("📊 R² Score on full dataset:", r2_score(y, y_pred))
    print("🔢 Sample predictions:", y_pred[:5])

if __name__ == "__main__":
    main()

