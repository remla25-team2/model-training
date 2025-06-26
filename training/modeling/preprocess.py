from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from lib_ml.preprocessing import TextPreprocessor

from training.config import DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR


app = typer.Typer()


@app.command()
def preprocess(
    dataset_path: Path = DATA_DIR / "raw" / "a1_RestaurantReviews_HistoricDump.tsv",
    corpus_path: Path = PROCESSED_DATA_DIR / "corpus.pkl",
    preprocessor_path: Path = MODELS_DIR / "bow" / "BoW_Sentiment_Model.pkl"
):
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path, delimiter="\t", quoting=3)

    preprocessor = TextPreprocessor()
    # Fit the vectorizer on the review text and transform
    corpus_matrix = preprocessor.fit(df['Review']).transform(df['Review'])
    logger.info("Preprocessing complete")

    logger.info(f"Saving cleaned corpus to {corpus_path}")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the sparse matrix as pickle
    pd.Series([corpus_matrix]).to_pickle(corpus_path)
    
    # Save the fitted preprocessor
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(preprocessor_path)
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")
    


if __name__ == "__main__":
    app()
