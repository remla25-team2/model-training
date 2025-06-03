from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from training.config import DATA_DIR, PROCESSED_DATA_DIR

from lib_ml.preprocessing import _clean

app = typer.Typer()


@app.command()
def preprocess(
    dataset_path: Path = DATA_DIR / "raw" / "a1_RestaurantReviews_HistoricDump.tsv",
    corpus_path: Path = PROCESSED_DATA_DIR / "corpus.pkl",
):
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path, delimiter="\t", quoting=3)

    corpus = []
    for i in tqdm(range(len(df)), desc="Cleaning text"):
        corpus.append(_clean(df["Review"][i]))
    logger.info("Preprocessing complete")

    logger.info(f"Saving cleaned corpus to {corpus_path}")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(corpus).to_pickle(corpus_path)


if __name__ == "__main__":
    app()
