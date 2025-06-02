# import pandas as pd
# from pathlib import Path

# def test_raw_data_schema():
#     data_path = Path("data/raw/a1_RestaurantReviews_HistoricDump.tsv")
#     df = pd.read_csv(data_path, delimiter='\t', quoting=3)
#     assert "Review" in df.columns
#     assert "Liked" in df.columns
#     assert df["Review"].notnull().all()
#     assert df["Liked"].isin([0, 1]).all()
#     assert df["Review"].str.strip().ne("").all()
#     assert df["Liked"].dtype in [int, "int64", "int32"]

# def test_no_duplicate_reviews():
#     data_path = Path("data/raw/a1_RestaurantReviews_HistoricDump.tsv")
#     df = pd.read_csv(data_path, delimiter='\t', quoting=3)
#     assert df["Review"].duplicated().sum() == 0
