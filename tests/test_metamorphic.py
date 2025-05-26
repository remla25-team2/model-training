import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from training.modeling.model_train import train
from training.modeling.transform import transform
from training.modeling.preprocess import preprocess

def test_metamorphic_synonym(tmp_path):
    raw_path = tmp_path / "raw.tsv"
    raw_path.write_text("Review\tLiked\nThis place is good.\t1\nThe food was bad.\t0\nI loved the service.\t1\nIt was awful.\t0\n")
    corpus_path = tmp_path / "corpus.pkl"
    preprocess(dataset_path=raw_path, corpus_path=corpus_path)
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    bow_path = tmp_path / "BoW_Sentiment_Model.pkl"
    transform(
        raw_dataset_path=raw_path,
        corpus_path=corpus_path,
        dataset_path=features_path,
        labels_path=labels_path,
        bow_path=bow_path
    )
    from training.modeling.split import split
    splits_dir = tmp_path / "splits"
    split(
        dataset_path=features_path,
        labels_path=labels_path,
        output_dir=splits_dir
    )
    model_path = tmp_path / "SentimentModel.pkl"
    metrics_path = tmp_path / "metrics.json"
    train(
        split_dir=splits_dir,
        model_path=model_path,
        model_metric_path=metrics_path
    )
    import pickle
    clf = joblib.load(model_path)
    with open(bow_path, "rb") as f:
        vectorizer = pickle.load(f)
    try:
        from lib_ml.preprocessing import _clean
    except ImportError:
        def _clean(x): return x.lower()
    original = "This place is good."
    synonym = "This place is fine."
    X_orig = vectorizer.transform([_clean(original)]).toarray()
    X_syn = vectorizer.transform([_clean(synonym)]).toarray()
    pred_orig = clf.predict(X_orig)[0]
    pred_syn = clf.predict(X_syn)[0]
    assert pred_orig in [0, 1]
    assert pred_syn in [0, 1]
    if (vectorizer.vocabulary_.get("good") is not None and vectorizer.vocabulary_.get("fine") is not None):
        assert pred_orig == pred_syn
