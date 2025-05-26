import pytest
from pathlib import Path
import subprocess
import json

PROJECT_ROOT = Path(__file__).parent.parent
EXPECTED_OUTPUTS = [
    PROJECT_ROOT / "data" / "processed" / "corpus.pkl",
    PROJECT_ROOT / "data" / "processed" / "features.csv",
    PROJECT_ROOT / "data" / "processed" / "labels.csv",
    PROJECT_ROOT / "data" / "processed" / "splits" / "X_train.csv",
    PROJECT_ROOT / "data" / "processed" / "splits" / "X_test.csv",
    PROJECT_ROOT / "data" / "processed" / "splits" / "y_train.csv",
    PROJECT_ROOT / "data" / "processed" / "splits" / "y_test.csv",
    PROJECT_ROOT / "models" / "bow" / "BoW_Sentiment_Model.pkl",
    PROJECT_ROOT / "models" / "SentimentModel.pkl",
    PROJECT_ROOT / "models" / "metrics.json",
]
REQUIRED_RAW_DATA = PROJECT_ROOT / "data" / "raw" / "a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def prepare_raw_data_for_pipeline_test():
    if not REQUIRED_RAW_DATA.exists():
        REQUIRED_RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        dummy_data = """Review\tLiked
Wow... Loved this place.\t1
Crust is not good.\t0
Not tasty and the texture was just nasty.\t0
Great service!\t1
Horrible experience.\t0
"""
        REQUIRED_RAW_DATA.write_text(dummy_data)
    yield

@pytest.mark.integration
def test_dvc_pipeline_repro(prepare_raw_data_for_pipeline_test):
    result = subprocess.run(
        ["dvc", "repro"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=600
    )
    for output_path in EXPECTED_OUTPUTS:
        assert output_path.exists(), f"Missing: {output_path}"
        if not output_path.is_dir():
            assert output_path.stat().st_size > 0, f"Empty: {output_path}"
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics
