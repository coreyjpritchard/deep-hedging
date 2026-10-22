# Rescore every hedger and redraw the figures from the cached weights. Needs uv.
.PHONY: reproduce sync test lint figures

reproduce: sync test figures

sync:
	uv sync --extra dev

test:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .

figures:
	uv run python -m deephedge.figures
