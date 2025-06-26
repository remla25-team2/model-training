# ML Infrastructure test

import pandas as pd
from pathlib import Path
from training.modeling.preprocess import preprocess

def test_preprocess_creates_corpus(tmp_path):
    raw_path = tmp_path / "dummy.tsv"
    raw_path.write_text("Review\tLiked\nThis is good.\t1\nBad food.\t0\n")
    corpus_path = tmp_path / "corpus.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    preprocess(dataset_path=raw_path, corpus_path=corpus_path, preprocessor_path=preprocessor_path)

    assert corpus_path.exists()
    corpus_series = pd.read_pickle(corpus_path)
    
    # The preprocess function saves a Series with one element (the sparse matrix)
    assert len(corpus_series) == 1, f"Expected 1 series element, got {len(corpus_series)}"
    
    # Get the actual corpus matrix
    corpus_matrix = corpus_series.iloc[0]
    
    # Check that it has the right shape (2 samples)
    assert corpus_matrix.shape[0] == 2, f"Expected 2 samples, got {corpus_matrix.shape[0]}"
    
    # Check that preprocessor was also saved
    assert preprocessor_path.exists()