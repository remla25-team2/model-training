# sentiment-analysis
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![Coverage](https://codecov.io/gh/remla25-team2/model-training/branch/main/graph/badge.svg)
![pylint]()


## DVC setup
This repository is using Data Version Control with remote storage. To pick files from remote storage do:

- ```make requirements``` to download all requirements and dvc
-  ```dvc pull``` this will pull all training model files from the Google Drive. Follow authenticqtion in browser to get access to the remote storage.
-  ```dvc push``` will push your changes to the model files to the remote storage if you have appropriate access.

To launch training pipeline:
- ```dvc repro``` it will launch all pipeline stages and create the missing files locally

To check model metrics:
- ```dvc exp show --no-pager``` will show the model accuracy, precision and recall

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         training and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── training   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes training a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## ML Test Score Coverage

| Category                | Test File                | Description                                 |
|-------------------------|-------------------------|---------------------------------------------|
| Feature & Data Integrity| test_data_integrity.py   | Schema, missing values          |
| Model Development       | test_model_train.py      | Model training           |
| ML Infrastructure       | test_pipeline.py         | DVC outputs            |
| Monitoring              | test_monitor.py          | Data drift               |
| Mutamorphic testing     | test_mutamorpic.py       | Mutamorphism             |
