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
    bow_path: Path = MODELS_DIR / "bow" / "BoW_Sentiment_Model.pkl",
):
    logger.info(f"Loading corpus from {corpus_path}")
    corpus = pd.read_pickle(corpus_path)

    logger.info("Creating bag-of-words features...")
    tp = TextPreprocessor(max_features=1420)
    X = tp.fit(corpus).transform(corpus)

    logger.info(f"Saving features to {dataset_path}")
    pd.DataFrame(X.toarray(), columns=tp.vectorizer.get_feature_names()).to_csv(dataset_path, index=False)


    # Labels: assumed to be in original corpus file name
    df = pd.read_csv(raw_dataset_path, delimiter="\t", quoting=3)
    y = df.iloc[:, -1].values
    pd.Series(y).to_csv(labels_path, index=False)

    tp.save(bow_path)
    logger.info(f"Saved TextPreprocessor (BoW model) to {bow_path}")
    # bow_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(bow_path, "wb") as f:
    #     pickle.dump(tp, f)


if __name__ == "__main__":
    app()
