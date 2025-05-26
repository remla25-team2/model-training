import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib
import json

from training.modeling.model_train import train

def test_model_train_end_to_end(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    X_train = pd.DataFrame(np.random.rand(10, 5))
    X_test = pd.DataFrame(np.random.rand(5, 5))
    y_train = pd.Series(np.random.randint(0, 2, size=10))
    y_test = pd.Series(np.random.randint(0, 2, size=5))
    X_train.to_csv(split_dir / "X_train.csv", index=False)
    X_test.to_csv(split_dir / "X_test.csv", index=False)
    y_train.to_csv(split_dir / "y_train.csv", index=False)
    y_test.to_csv(split_dir / "y_test.csv", index=False)

    model_path = tmp_path / "SentimentModel.pkl"
    metrics_path = tmp_path / "metrics.json"

    train(
        split_dir=split_dir,
        model_path=model_path,
        model_metric_path=metrics_path
    )

    assert model_path.exists()
    clf = joblib.load(model_path)
    assert hasattr(clf, "predict")

    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics = json.load(f)
    for key in ["accuracy", "precision", "recall", "confusion_matrix"]:
        assert key in metrics
