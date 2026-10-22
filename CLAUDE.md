# deep-hedging: conventions

Public repository, a personal research project in mathematical finance. The package is `deephedge`. Apps in
`sessions/` and tests in `tests/` import from it and hold no logic of their own. Current
state and next session are at the top of `LOG.md`.

## Method

- One question per session, one sitting, one marimo app in `sessions/`. Figure and sliders
  first, explanation below, proofs only where the proof is the point.
- Levels: 0 play (a slider and a guess), 1 reproduce (a figure from a paper), 2 implement
  (written by hand, tests check it), 3 extend (change one assumption, log the result).
- Level 3 results are exploratory: report effect size, direction and the next specification.
  Never conclude that nothing is there.

## Writing

- Lead with the picture and what it shows. Then the definition. Then the result and its source.
- Plain, literal prose. British spelling. No em dashes or en dashes.
- A paragraph's main point is its first sentence. No drama in the last line.

## Code

- Paths are arrays of shape `(m, n + 1)` including the `t = 0` column.
- Every random function takes `seed: int` and uses `np.random.default_rng(seed)`.
- Statistical tests state tolerance in standard errors and the sample size behind it.
- `ruff check` clean, line length 100, py311. torch on CPU; training under five minutes.
- Install editable into a Python 3.11 virtual environment: `pip install -e ".[dev]"`.

## Git

- Commit directly to `main`. A commit per session, message `sNN: <what it established>`.
- Tag `sNN` once the session has been worked through.
