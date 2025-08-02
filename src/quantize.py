import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_model(path='model.joblib'):
    return joblib.load(path)

def quantize_params(coef, intercept):
    coef_min, coef_max = coef.min(), coef.max()
    intercept_min, intercept_max = intercept.min(), intercept.max()

    coef_scaled = ((coef - coef_min) / (coef_max - coef_min) * 255).astype(np.uint8)
    intercept_scaled = ((intercept - intercept_min) / (intercept_max - intercept_min) * 255).astype(np.uint8)

    scaling_factors = {
        "coef_min": coef_min,
        "coef_max": coef_max,
        "intercept_min": intercept_min,
        "intercept_max": intercept_max
    }

    return coef_scaled, intercept_scaled, scaling_factors

def dequantize_params(q_coef, q_intercept, scale):
    coef = q_coef.astype(np.float32) / 255 * (scale["coef_max"] - scale["coef_min"]) + scale["coef_min"]
    intercept = q_intercept.astype(np.float32) / 255 * (scale["intercept_max"] - scale["intercept_min"]) + scale["intercept_min"]
    return coef, intercept

def inference(X, coef, intercept):
    return np.dot(X, coef) + intercept

if __name__ == "__main__":
    # Load trained model
    model = load_model()
    coef, intercept = model.coef_, np.array([model.intercept_])

    # Save unquantized parameters
    joblib.dump((coef, intercept), "unquant_params.joblib")

    # Quantize
    q_coef, q_intercept, scale = quantize_params(coef, intercept)
    joblib.dump((q_coef, q_intercept, scale), "quant_params.joblib")

    # Dequantize
    dq_coef, dq_intercept = dequantize_params(q_coef, q_intercept, scale)

    # Run inference using dequantized weights
    data = fetch_california_housing()
    X = data.data
    y_pred = inference(X, dq_coef, dq_intercept)

    print("✅ Quantization done.")
    print("🔢 Sample predictions:", y_pred[:5])

