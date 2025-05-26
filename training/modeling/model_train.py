import json
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import typer

from training.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def train(
    split_dir: Path = PROCESSED_DATA_DIR / "splits",
    model_path: Path = MODELS_DIR / "SentimentModel.pkl",
    model_metric_path: Path = MODELS_DIR / "metrics.json",
):
    logger.info("Loading training and test data...")
    X_train = pd.read_csv(split_dir / "X_train.csv").values
    X_test = pd.read_csv(split_dir / "X_test.csv").values
    y_train = pd.read_csv(split_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(split_dir / "y_test.csv").values.ravel()

    logger.info("Training Gaussian Naive Bayes model...")
    clf = GaussianNB()

    with tqdm(total=1, desc="Training") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    logger.success(f"Accuracy: {acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
    }

    with open(model_metric_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    app()
