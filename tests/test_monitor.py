"""
Performance Monitoring Tests
"""

import time
import numpy as np
from memory_profiler import memory_usage

# Example input to simulate a realistic prediction
SAMPLE_INPUT = np.random.rand(1, 1421)

# Thresholds – Adjust based on actual benchmarks
MAX_MEMORY_MB = 500         # Maximum acceptable memory usage
MAX_LATENCY_MS = 1          # Maximum time for a single prediction
MIN_THROUGHPUT = 1000       # Minimum acceptable predictions per second


def test_memory_usage_during_prediction(trained_model):
    """
    Ensure memory usage during a single prediction stays within limits.
    """
    def run_prediction():
        trained_model.predict(SAMPLE_INPUT)

    peak_memory = max(memory_usage(run_prediction, interval=0.1, timeout=1))
    print(f"[Memory] Peak usage: {peak_memory:.1f} MB")

    assert peak_memory <= MAX_MEMORY_MB, (
        f"Exceeded memory limit: {peak_memory:.1f} MB used, limit is {MAX_MEMORY_MB} MB"
    )


def test_prediction_latency(trained_model):
    """
    Ensure prediction latency does not exceed the defined threshold.
    """
    start = time.perf_counter()
    trained_model.predict(SAMPLE_INPUT)
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"[Latency] Prediction time: {latency_ms:.1f} ms")

    assert latency_ms <= MAX_LATENCY_MS, (
        f"Prediction too slow: {latency_ms:.1f} ms > {MAX_LATENCY_MS} ms"
    )


def test_model_throughput(trained_model):
    """
    Ensure model can sustain a high enough throughput.
    """
    total_runs = 1000
    start = time.perf_counter()

    for _ in range(total_runs):
        trained_model.predict(SAMPLE_INPUT)

    elapsed = time.perf_counter() - start
    throughput = total_runs / elapsed
    print(f"[Throughput] {throughput:.1f} predictions/sec")

    assert throughput >= MIN_THROUGHPUT, (
        f"Throughput too low: {throughput:.1f} < {MIN_THROUGHPUT} predictions/sec"
    )
