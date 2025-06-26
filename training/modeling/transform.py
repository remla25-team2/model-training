from pathlib import Path

from loguru import logger
import pandas as pd
import typer
from lib_ml.preprocessing import TextPreprocessor
from training.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def transform(
    raw_dataset_path: Path = DATA_DIR / "raw" / "a1_RestaurantReviews_HistoricDump.tsv",
    corpus_path: Path = PROCESSED_DATA_DIR / "corpus.pkl",
    dataset_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    preprocessor_path: Path = MODELS_DIR / "bow" / "BoW_Sentiment_Model.pkl"
):
    logger.info(f"Loading corpus from {corpus_path}")
    corpus_series = pd.read_pickle(corpus_path)
    corpus_matrix = corpus_series.iloc[0]

    logger.info(f"Loading fitted preprocessor from {preprocessor_path}")
    tp = TextPreprocessor.load(preprocessor_path)

    logger.info(f"Saving features to {dataset_path}")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    # pylint: disable=W0212
    feature_names = tp._vectorizer.get_feature_names_out()
    pd.DataFrame(corpus_matrix.toarray(), columns=feature_names).to_csv(dataset_path, index=False)
    # pylint: enable=W0212
    
    logger.info(f"Saving labels to {labels_path}")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_dataset_path, delimiter="\t", quoting=3)
    y = df.iloc[:, -1].values
    pd.Series(y).to_csv(labels_path, index=False)
    logger.info("Transformation complete.")

if __name__ == "__main__":
    app()
