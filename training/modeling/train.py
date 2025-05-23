from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from training.config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR

# Model training imports
import numpy as np
import pandas as pd
import pickle
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from lib_ml.preprocessing import _clean

app = typer.Typer()

class SentimentAnalysisModel:
    def __init__(self, dataset_path):
        logger.info(f"Loading dataset from {dataset_path}")
        self.dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
        logger.info(f"Dataset loaded with {len(self.dataset)} records")

    def preprocess(self):
        logger.info("Preprocessing text data...")
        corpus = []
        for i in tqdm(range(len(self.dataset)), desc="Cleaning text"):
            corpus.append(_clean(self.dataset['Review'][i]))
        logger.info("Preprocessing complete")
        return corpus

    def transform(self, corpus, bow_dir):
        logger.info("Transforming text to features...")
        os.makedirs(bow_dir, exist_ok=True)
        bow_path = bow_dir / "BoW_Sentiment_Model.pkl"

        cv = CountVectorizer(max_features=1420)
        X = cv.fit_transform(corpus).toarray()
        y = self.dataset.iloc[:, -1].values
        
        logger.info(f"Saving bag-of-words vectorizer to {bow_path}")
        with open(bow_path, "wb") as f:
            pickle.dump(cv, f)

        logger.info(f"Generated feature matrix with shape {X.shape}")
        return X, y

    def splitdata(self, x, y):
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test, model_path):
        logger.info("Training Gaussian Naive Bayes classifier...")
        classifier = GaussianNB()
        
        with tqdm(total=1, desc="Training model") as pbar:
            classifier.fit(X_train, y_train)
            pbar.update(1)
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(classifier, model_path)
        
        # Optional: evaluate model
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")
        
        return classifier, accuracy

@app.command()
def main(
    dataset_path: Path = DATA_DIR / "raw" / "a1_RestaurantReviews_HistoricDump.tsv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "SentimentModel.pkl",
    bow_dir: Path = MODELS_DIR / "bow"
):
    """
    Train a sentiment analysis model using the specified dataset.
    
    The model and bag-of-words vectorizer will be saved to the specified paths.
    """
    logger.info("Starting sentiment analysis model training...")
    
    # Initialize model with dataset
    sentiment = SentimentAnalysisModel(dataset_path=dataset_path)
    
    # Preprocess text data
    corpus = sentiment.preprocess()
    
    # Transform text to features
    x, y = sentiment.transform(corpus, bow_dir)
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = sentiment.splitdata(x, y)
    
    # Train model and save
    model, accuracy = sentiment.train(X_train, X_test, y_train, y_test, model_path)
    
    logger.success(f"Model training complete with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    app()