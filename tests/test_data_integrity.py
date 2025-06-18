#Feature & Data test
import pandas as pd
from pathlib import Path

def test_raw_data_schema():
    data_path = Path("data/raw/a1_RestaurantReviews_HistoricDump.tsv")
    assert data_path.exists(), f"Data file not found: {data_path}"

    df = pd.read_csv(data_path, delimiter='\t', quoting=3)

    required_columns = {"Review", "Liked"}
    assert required_columns.issubset(df.columns), f"Missing columns: {required_columns - set(df.columns)}"

    assert df["Review"].notnull().all()
    assert df["Liked"].isin([0, 1]).all()

    assert df["Review"].str.strip().ne("").all()
    assert (df["Review"].str.len() > 0).all(), "Empty reviews found"
    assert df["Liked"].dtype in [int, "int64", "int32"]