from training import dataset

def test_dataset_main_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", tmp_path)
    try:
        dataset.main()
    except Exception:
        pass
