# Rescore every hedger and redraw the figures from the cached weights. Needs uv.
.PHONY: reproduce sync test lint figures surfaces-train surfaces-eval

reproduce: sync test figures

sync:
	uv sync --extra dev

test:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .

figures:
	uv run python -m deephedge.figures

# Train and evaluate snapshot-calibrated hedgers. CALIBRATION= is a local
# training document; outputs under data/weights/surfaces/ and
# data/exports/local/ are gitignored and are not committed.
surfaces-train:
	@if [ -z "$(CALIBRATION)" ]; then echo "surfaces-train needs CALIBRATION=..."; exit 2; fi
	uv run python -m deephedge.surfaces train --calibration "$(CALIBRATION)"

surfaces-eval:
	@if [ -z "$(CALIBRATION)" ]; then echo "surfaces-eval needs CALIBRATION=..."; exit 2; fi
	uv run python -m deephedge.surfaces eval --calibration "$(CALIBRATION)" --out data/exports/local/surfaces_eval.json
