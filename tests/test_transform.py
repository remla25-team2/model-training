# Feature & Data test
import pandas as pd
from pathlib import Path
from training.modeling.transform import transform

def test_transform_creates_outputs(tmp_path):
    raw_path = tmp_path / "dummy.tsv"
    raw_path.write_text("Review\tLiked\nThis is good.\t1\nBad food.\t0\n")
    
    # Create and fit a preprocessor for our test data
    from lib_ml.preprocessing import TextPreprocessor
    test_reviews = ["This is good.", "Bad food."]
    preprocessor = TextPreprocessor()
    corpus_matrix = preprocessor.fit(test_reviews).transform(test_reviews)
    
    # Save the corpus and preprocessor
    corpus_path = tmp_path / "corpus.pkl"
    pd.Series([corpus_matrix]).to_pickle(corpus_path)
    
    bow_path = tmp_path / "BoW_Sentiment_Model.pkl"
    bow_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(bow_path)

    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"

    transform(
        raw_dataset_path=raw_path,
        corpus_path=corpus_path,
        dataset_path=features_path,
        labels_path=labels_path,
        preprocessor_path=bow_path,
    )

    assert features_path.exists()
    assert labels_path.exists()
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    assert not features.empty
    assert not labels.empty
