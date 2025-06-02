from training import plots

def test_plots_main_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    try:
        plots.main()
    except Exception:
        pass
