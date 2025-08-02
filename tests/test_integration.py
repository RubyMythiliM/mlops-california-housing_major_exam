import os
import joblib
from sklearn.linear_model import LinearRegression

def test_model_file_exists():
    model_path = os.path.join("artifacts", "model.joblib")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    model = joblib.load(model_path)
    assert isinstance(model, LinearRegression)

