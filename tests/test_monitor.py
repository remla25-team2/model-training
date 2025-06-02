"""
Performance Monitoring Tests
"""

import time
import numpy as np
from memory_profiler import memory_usage
from pathlib import Path
import joblib
import pytest

# Example input to simulate a realistic prediction
SAMPLE_INPUT = np.random.rand(1, 13)

# Thresholds – Adjust based on actual benchmarks
MAX_MEMORY_MB = 500         # Maximum acceptable memory usage
MAX_LATENCY_SEC = 0.1          # Maximum time for a single prediction
MIN_THROUGHPUT = 1000       # Minimum acceptable predictions per second


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """
    Fixture that loads a pre‐trained GaussianNB model from disk.
    Adjust the path below if your model is stored elsewhere.
    """
    # Assume the trained model was saved via joblib.dump(...)
    model_path = Path("models") / "SentimentModel.pkl"
    if not model_path.exists():
        pytest.skip(f"Trained model not found at {model_path}")
    model = joblib.load(model_path)
    return model


def test_memory_usage_during_prediction(trained_model):
    """
    Ensure memory usage during a single prediction stays within limits.
    """

    def run_prediction():
        trained_model.predict(SAMPLE_INPUT)

    # Measure memory usage (MB) while running run_prediction()
    peak_memory = max(memory_usage(run_prediction, interval=0.1, timeout=1))
    assert peak_memory < MAX_MEMORY_MB, (
        f"Peak memory {peak_memory:.1f} MB exceeds limit of {MAX_MEMORY_MB} MB."
    )


def test_prediction_latency(trained_model):
    """
    Ensure prediction latency does not exceed the defined threshold.
    """
    start = time.perf_counter()
    _ = trained_model.predict(SAMPLE_INPUT)
    latency = time.perf_counter() - start
    assert latency < MAX_LATENCY_SEC, (
        f"Prediction latency {latency:.4f}s exceeds {MAX_LATENCY_SEC}s."
    )


def test_model_throughput(trained_model):
    """
    Ensure model can sustain a high enough throughput.
    """
    total_runs = 1000
    start = time.perf_counter()
    for _ in range(total_runs):
        _ = trained_model.predict(SAMPLE_INPUT)
    duration = time.perf_counter() - start
    throughput = total_runs / duration
    assert throughput > MIN_THROUGHPUT, (
        f"Throughput {throughput:.1f} req/s below minimum {MIN_THROUGHPUT} req/s."
    )


