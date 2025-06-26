#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = model-training
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install flit
	flit install --pth-file
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 training
	isort --check --diff training
	black --check training

## Advanced linting with pylint and bandit
.PHONY: lint-advanced
lint-advanced:
	pylint training
	bandit -c bandit.yaml -r ./training || true

## All linting checks
.PHONY: lint-all
lint-all: lint lint-advanced

## Format source code with black
.PHONY: format
format:
	isort training
	black training

## Run tests with coverage
.PHONY: test
test:
	python -m pytest tests --cov=training --cov-report=term-missing --cov-report=xml

## Generate quality reports
.PHONY: quality-report
quality-report:
	@echo "=== Code Quality Report ==="
	@echo "Running pylint..."
	@pylint training --output-format=text --score=yes | tee pylint-report.txt || true
	@echo "Running flake8..."
	@flake8 training --statistics --tee --output-file=flake8-report.txt || true
	@echo "Running bandit..."
	@bandit -r training -f json -o bandit-report.json || true
	@echo "Running tests with coverage..."
	@python -m pytest tests --cov=training --cov-report=term-missing --cov-report=xml || true
	@echo "Quality reports generated in: pylint-report.txt, flake8-report.txt, bandit-report.json, coverage.xml"
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	gsutil -m rsync -r gs://remla25-team2/data/ data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	gsutil -m rsync -r data/ gs://remla25-team2/data/
	



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) training/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
