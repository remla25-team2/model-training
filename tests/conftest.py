# tests/conftest.py
import pytest
import joblib
from pathlib import Path

@pytest.fixture(scope="session")
def trained_model():
    model_path = Path("models/SentimentModel.pkl")
    if not model_path.exists():
        pytest.skip("Trained model not found. Run the pipeline first.")
    model = joblib.load(model_path)
    return model

@pytest.fixture(scope="session")
def test_data():
    import numpy as np
    X = np.random.rand(10, 1420)
    return {"X": X}
