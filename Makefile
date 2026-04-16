.PHONY: install test test-cov lint format run-baseline run-ablation clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/unit -v

test-cov:
	pytest tests/unit --cov=src --cov=experiments --cov-report=term

lint:
	python -m compileall -q src experiments

format:
	@echo "Format hooks are not configured yet"

run-baseline:
	python experiments/exp_baseline.py --config configs/default.yaml

run-ablation:
	python experiments/exp_ablation.py --config configs/default.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
