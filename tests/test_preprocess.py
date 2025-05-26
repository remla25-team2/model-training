import pandas as pd
from pathlib import Path
from training.modeling.preprocess import preprocess

def test_preprocess_creates_corpus(tmp_path):
    raw_path = tmp_path / "dummy.tsv"
    raw_path.write_text("Review\tLiked\nThis is good.\t1\nBad food.\t0\n")
    corpus_path = tmp_path / "corpus.pkl"

    preprocess(dataset_path=raw_path, corpus_path=corpus_path)

    assert corpus_path.exists()
    corpus = pd.read_pickle(corpus_path)
    assert len(corpus) == 2
    assert all(isinstance(x, str) for x in corpus)
