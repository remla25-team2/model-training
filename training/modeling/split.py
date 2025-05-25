from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from training.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def split(
    dataset_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    output_dir: Path = PROCESSED_DATA_DIR / "splits",
):
    logger.info("Loading features and labels...")
    X = pd.read_csv(dataset_path).values
    y = pd.read_csv(labels_path).values.ravel()

    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train).to_csv(output_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(output_dir / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(output_dir / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(output_dir / "y_test.csv", index=False)

    logger.info(f"Saved split datasets to {output_dir}")


if __name__ == "__main__":
    app()
