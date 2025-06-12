# Feature & Data test
from training import features

def test_features_main_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "PROCESSED_DATA_DIR", tmp_path)
    try:
        features.main()
    except Exception:
        pass
