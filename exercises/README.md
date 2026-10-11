# Exercises

Corey's own implementations. Run the shared tests against them with
`pytest --impl=exercises`. The equals sign matters: `exercises/` is a directory,
and pytest would otherwise treat that word as a path. Untracked by default.

Use the module and function names of `deephedge` (for example `exercises/hedging.py`
with `delta_hedge_pnl`, `exercises/env.py` with `BlackScholes`, or
`exercises/heston.py` with `Heston`). Tests that need a function you have not
written yet are skipped.
