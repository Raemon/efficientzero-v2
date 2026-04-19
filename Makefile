.PHONY: install install-atari install-dmc install-dev smoke test format train-atari train-dmc-proprio train-dmc-vision

install:
	pip install -e .

install-atari:
	pip install -e ".[atari]"

install-dmc:
	pip install -e ".[dmc]"

install-dev:
	pip install -e ".[dev]"

smoke:
	python scripts/train.py --config configs/smoke.yaml

test:
	pytest -q tests/

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

train-atari:
	python scripts/train.py --config configs/atari.yaml

train-dmc-proprio:
	python scripts/train.py --config configs/dmc_proprio.yaml

train-dmc-vision:
	python scripts/train.py --config configs/dmc_vision.yaml
