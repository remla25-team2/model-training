import pandas as pd
from pathlib import Path

def test_data_drift():
    processed = Path("data/processed/features.csv")
    if not processed.exists():
        import subprocess
        subprocess.run(["dvc", "repro"], check=True)
    df = pd.read_csv(processed)
    means = df.mean()
    stds = df.std()
    assert not means.isnull().any()
    assert not stds.isnull().any()
    assert (stds > 0).all()
