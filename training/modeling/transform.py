import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from loguru import logger
import typer
from training.config import PROCESSED_DATA_DIR, MODELS_DIR, DATA_DIR

app = typer.Typer()

@app.command()
def transform(
    raw_dataset_path: Path = DATA_DIR / "raw" / "a1_RestaurantReviews_HistoricDump.tsv",
    corpus_path: Path = PROCESSED_DATA_DIR / "corpus.pkl",
    dataset_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    bow_path: Path = MODELS_DIR / "bow" / "BoW_Sentiment_Model.pkl",
):
    logger.info(f"Loading corpus from {corpus_path}")
    corpus = pd.read_pickle(corpus_path)

    logger.info("Creating bag-of-words features...")
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()

    logger.info(f"Saving features to {dataset_path}")
    feature_names = cv.get_feature_names_out()
    pd.DataFrame(X, columns=feature_names).to_csv(dataset_path, index=False)

    # Labels: assumed to be in original corpus file name
    df = pd.read_csv(raw_dataset_path, delimiter="\t", quoting=3)
    y = df.iloc[:, -1].values
    pd.Series(y).to_csv(labels_path, index=False)

    bow_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bow_path, "wb") as f:
        pickle.dump(cv, f)

if __name__ == "__main__":
    app()