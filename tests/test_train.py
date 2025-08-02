import pytest
from sklearn.linear_model import LinearRegression
import joblib
import os
from src.train import load_data, train_model, evaluate_model

def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_instance():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)

def test_model_trained():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')

def test_model_accuracy():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    r2, _ = evaluate_model(model, X_test, y_test)
    assert r2 > 0.5

def test_model_file_exists():
    assert os.path.exists("src/model.joblib")
    model = joblib.load("src/model.joblib")
    assert isinstance(model, LinearRegression)

