# AGENTS.md — FedTracker-Pro

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

## Commands

| Task | Command |
|------|---------|
| Lint (syntax check only) | `make lint` — runs `python -m compileall -q src experiments` |
| Unit tests | `make test` — runs `pytest tests/unit -v` |
| Integration tests | `pytest tests/integration -v` |
| All tests | `pytest tests/ -v` |
| Coverage | `make test-cov` — `--cov=src --cov=experiments` |
| Run baseline experiment | `make run-baseline` |
| Run ablation experiment | `make run-ablation` |

**Note:** `make lint` is compile-only (`compileall`). No formatter, linter, or type-checker is wired up in the Makefile. `black`, `flake8`, `isort`, `mypy` are in `requirements-dev.txt` but have no `make` targets.

## Architecture

- **Language:** Python 3.10+, PyTorch 2.0+
- **Packages:** `src` (main library) and `experiments` (experiment runners) — both are installable via `setup.py`
- **Entry point:** `src.core.fed_tracker_pro.FedTrackerPro` — orchestrates federated training, defense, verification, and attack evaluation
- **Config:** `src.core.config.Config` loads from YAML. Config is split into dataclasses: `federated`, `data`, `model`, `watermark`, `fingerprint`, `adaptive` (YAML key `adaptive_allocation`), `crypto`, `unlearning`, `verification`, `system`, `experiment`
- **Default config:** `configs/default.yaml`

### Key source directories

- `src/core/` — framework entry point, client, server, config
- `src/aggregation/` — FedAvg, FedProx aggregators
- `src/defense/` — watermark, fingerprint, adaptive allocation, crypto verification, unlearning-guided relocation, multi-layer verification
- `src/attacks/` — fine-tuning, pruning, quantization, overwriting, ambiguity, model extraction
- `src/datasets/` — federated data manager, CIFAR/MNIST adapters
- `src/models/` — ResNet, VGG, MobileNetV2, DiffusionUNet
- `src/utils/` — logger, metrics, data utils, crypto utils, visualization
- `experiments/` — experiment scripts (`exp_baseline.py`, `exp_ablation.py`, `exp_robustness.py`, `exp_scalability.py`) and their configs under `experiments/configs/`

### Defense modules are independently toggleable

Each defense module has an `enabled` flag in config: `watermark`, `fingerprint`, `adaptive_allocation`, `crypto`, `unlearning`.

## Testing

- 175 unit tests in `tests/unit/`, 3 integration tests in `tests/integration/`
- Tests use CPU by default (no GPU required)
- `tests/fixtures/` exists but is currently empty

## Conventions

- Code comments and docstrings are in Chinese (中文)
- Config YAML keys use `snake_case`
- The `adaptive` config dataclass maps to the `adaptive_allocation` YAML key (both `adaptive_allocation` and `adaptive` are accepted in YAML)
- Models are instantiated via factory functions: `ResNet18(num_classes)`, `VGG16(num_classes)`, etc.
- `DiffusionUNet` requires even spatial dimensions in input
- Commit messages follow conventional commits style: `feat:`, `fix:`, etc.
