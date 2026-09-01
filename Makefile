lint:
	uv run ruff check src tests && uv run ruff format --check src tests
format:
	uv run ruff format src tests && uv run ruff check --fix src tests
test:
	uv run pytest -q
test-slow:
	uv run pytest -q -m slow tests/integration
benchmark:
	uv run deconvolute benchmark --datasets 3 --out reports/benchmark.md
train:
	uv run deconvolute train --datasets 8 --out models/graph_lr.json
train-tse:
	uv run deconvolute train-tse --epochs 30 --out models/tse/model.pt
.PHONY: lint format test test-slow benchmark train train-tse
