# FedTracker-Pro Tutorial

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## 2) Verify

```bash
pytest tests/unit -q
make lint
```

## 3) Baseline experiment

```bash
python experiments/exp_baseline.py --config configs/default.yaml
```

## 4) Ablation experiment

```bash
python experiments/exp_ablation.py --config configs/default.yaml
```

## 5) Use helper scripts

```bash
bash scripts/setup.sh
bash scripts/train.sh configs/default.yaml
bash scripts/evaluate.sh configs/default.yaml
```
