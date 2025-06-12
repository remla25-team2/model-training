# Feature & Data test

import pandas as pd
import numpy as np
from pathlib import Path
from training.modeling.split import split

def test_split_creates_files(tmp_path):
    features = pd.DataFrame(np.random.rand(20, 4))
    labels = pd.Series(np.random.randint(0, 2, size=20))
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)
    output_dir = tmp_path / "splits"

    split(dataset_path=features_path, labels_path=labels_path, output_dir=output_dir)

    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        f = output_dir / fname
        assert f.exists()
        df = pd.read_csv(f)
        assert not df.empty
