# Monitoring test
import time
import numpy as np
import pytest
from memory_profiler import memory_usage
import joblib
from pathlib import Path
import pandas as pd
from training.modeling.transform import transform as run_transform_pipeline
from training.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR


MAX_MEMORY_MB = 205         # Maximum allowed memory (MB) during a single predict
MAX_LATENCY_SEC = 0.1      # Maximum allowed latency (seconds) for one prediction
MIN_THROUGHPUT = 100       # Minimum allowed throughput (requests/sec)
MAX_FEATURE_EXTRACTION_TIME_SEC = 5.0 # Max time for feature extraction
MAX_FEATURE_EXTRACTION_MEMORY_MB = 500 # Max memory for feature extraction


@pytest.fixture(scope="module")
def trained_model():
    """
    Fixture that loads a pre‐trained GaussianNB model from disk,
    then yields it to tests.
    """
    model_path = Path("models") / "SentimentModel.pkl"
    if not model_path.exists():
        pytest.skip(f"Trained model not found at {model_path}")
    model = joblib.load(model_path)
    return model


def test_memory_usage_during_prediction(trained_model):
    """
    Ensure memory usage during a single prediction stays within limits.
    """
    # Read how many input features the model expects:
    n_features = trained_model.n_features_in_
    # Build a single random sample of shape (1, n_features)
    sample = np.random.rand(1, n_features)

    def run_prediction():
        trained_model.predict(sample)

    # Measure memory usage (in MB) while running run_prediction()
    peak_memory = max(memory_usage(run_prediction, interval=0.1, timeout=1))
    assert peak_memory < MAX_MEMORY_MB, (
        f"Peak memory {peak_memory:.1f} MB exceeds limit of {MAX_MEMORY_MB} MB."
    )


def test_prediction_latency(trained_model):
    """
    Ensure prediction latency does not exceed the defined threshold.
    """
    n_features = trained_model.n_features_in_
    sample = np.random.rand(1, n_features)

    start = time.perf_counter()
    _ = trained_model.predict(sample)
    latency = time.perf_counter() - start
    assert latency < MAX_LATENCY_SEC, (
        f"Prediction latency {latency:.4f}s exceeds {MAX_LATENCY_SEC}s."
    )


def test_model_throughput(trained_model):
    """
    Ensure model can sustain a high enough throughput.
    """
    n_features = trained_model.n_features_in_
    sample = np.random.rand(1, n_features)

    total_runs = 1000
    start = time.perf_counter()
    for _ in range(total_runs):
        _ = trained_model.predict(sample)
    duration = time.perf_counter() - start
    throughput = total_runs / duration
    assert throughput > MIN_THROUGHPUT, (
        f"Throughput {throughput:.1f} req/s below minimum {MIN_THROUGHPUT} req/s."
    )


def test_feature_extraction_cost():
    """
    Ensure feature extraction (transform stage) performance meets requirements.
    Measures both time and memory.
    """
    raw_path = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
    corpus_path = PROCESSED_DATA_DIR / "corpus.pkl"
    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"
    preprocessor_path = MODELS_DIR / "bow" / "BoW_Sentiment_Model.pkl"

    if not raw_path.exists():
        pytest.skip(f"Raw data not found at {raw_path}. Run 'dvc pull' first.")
    if not corpus_path.exists():
        pytest.skip(f"Corpus not found at {corpus_path}. Run 'dvc repro preprocess' first.")

    def measure_transform():
        run_transform_pipeline(
            raw_dataset_path=raw_path,
            corpus_path=corpus_path,
            dataset_path=features_path,
            labels_path=labels_path,
            preprocessor_path=preprocessor_path
        )

    # Measure time
    start_time = time.perf_counter()
    peak_memory_mb = max(memory_usage(measure_transform, interval=0.1, timeout=60))
    end_time = time.perf_counter()
    duration = end_time - start_time

    assert duration < MAX_FEATURE_EXTRACTION_TIME_SEC, \
        f"Feature extraction time {duration:.2f}s exceeds {MAX_FEATURE_EXTRACTION_TIME_SEC}s."
    assert peak_memory_mb < MAX_FEATURE_EXTRACTION_MEMORY_MB, \
        f"Feature extraction peak memory {peak_memory_mb:.2f}MB exceeds {MAX_FEATURE_EXTRACTION_MEMORY_MB}MB."

