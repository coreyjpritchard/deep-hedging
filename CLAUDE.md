# deep-hedging: instructions for builders

Public repository, part of Corey's PhD preparation. Shared method, file layout, writing and
code rules are in `~/phd/CONVENTIONS.md`. Read it first. The shared environment is
`~/phd/.venv` (activate it; do not create another).

The package is `deephedge`. Apps in `sessions/` and tests in `tests/` import from it and
hold no logic of their own. Current state and next session are at the top of `LOG.md`.
