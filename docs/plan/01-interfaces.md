# Interfaces: the shared typed contract

Plan only. Names and signatures are the target for M01, M02 and M05; modules may rename
with a note in the PR but must keep the shapes and the seed rules.

Conventions kept from `CLAUDE.md` and PR 1:

- Time grid `time (n+1,)` from 0 to maturity; column 0 is `t = 0`.
- Prices `instruments (m, n+1, d)`; positions `(m, n, d)` held over `[t_k, t_{k+1}]`; the
  book is flat before `t_0` and after maturity; opening and liquidation are charged.
- Everything in the P&L is in time-0 money; `discount (n+1,)` or `(m, n+1)`.
- Every random function takes `seed: int`; one documented draw order per generator.
- numpy for evaluation, torch for training; the torch P&L must equal the numpy P&L to
  1e-10 in float64 on the same positions.

## Paths

```python
@dataclass(frozen=True)
class Paths:
    time: np.ndarray            # (n+1,)
    instruments: np.ndarray     # (m, n+1, d)  tradable prices, enter the P&L
    features: np.ndarray        # (m, n+1, f)  observable by a hedger, not traded
    discount: np.ndarray        # (n+1,) or (m, n+1)
    state: Mapping[str, np.ndarray] = {}   # named extras: "variance", "regime",
                                           # "regime_prob", "xi", "realised_variance"
    instrument_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    source: object | None = None           # pass-through of the generator's native
                                           # object, e.g. volsurface MarketPaths (D4)
    def spot(self) -> np.ndarray            # instruments[:, :, 0]
    def to_torch(self, dtype=torch.float32) -> "TorchPaths"
```

Rule: instrument column 0 is the spot. Feature column 0 is the spot unless a feature
block replaces the feature array entirely; the validator checks `feature_names`.

## Generator (replaces `Market` as the simulator half)

```python
class Generator(Protocol):
    name: str
    maturity: float
    instrument_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    state_names: tuple[str, ...]
    def simulate(self, m: int, n: int, seed: int) -> Paths: ...
    def params(self) -> dict                   # serialisable, goes in the spec
    def pricer(self) -> "Pricer | None"        # None means Monte Carlo fallback only
```

Concrete generators: `BlackScholes`, `Heston` (from PR 1, made to conform), `SABR`,
`LocalVol` (adapter over `volsurface.models.localvol.LocalVol`), `RoughBergomi` (adapter
over `roughvol.rbergomi.RoughBergomi`: `Z * forward` as the spot, `state["v"]` as
variance, optional `state["xi"]`), `RegimeBergomi` (adapter over
`volsurface.markets.RegimeSwitchingBergomi`: spot, variance, `regime`, `regime_prob`,
`xi`, with variance-swap marks as a second instrument when asked; the `MarketPaths`
object is passed through in `Paths.source`, decision D4), `SPXSurface`
(snapshot to SANOS or SVI surface to Dupire grid to `LocalVol`, or to a calibrated rough
Bergomi), `UploadedPaths` and `BlockBootstrap` (from PR 4), `UserModule` (from PR 4
`upload.py`, local use only).

## Contract

```python
class Contract(Protocol):
    name: str
    def payoff(self, paths: Paths) -> np.ndarray          # (m,), undiscounted, at maturity
    def payoff_torch(self, paths: TorchPaths) -> torch.Tensor
    def monitoring(self) -> Monitoring                     # which grid points it reads
    def params(self) -> dict
```

Concrete contracts: `Vanilla(kind, strike)`, `Digital(kind, strike, cash)`,
`Barrier(kind, strike, barrier, direction, knock, monitoring="grid")`,
`Asian(kind, strike, averaging="arithmetic"|"geometric", dates)`,
`Lookback(kind, floating|fixed)`, `VarianceSwap(strike, realised="log")`,
`Basket(weights, kind, strike)` (needs a multi-asset generator), `ForwardStart(kind,
start, life)` (needed by the regime experiment's L2), `UserPayoff(expr)` where `expr` is a
restricted expression over `S`, `S_T`, `max_S`, `min_S`, `mean_S`, `RV` compiled to both
numpy and torch (no `exec` of arbitrary code in this block).

## Pricer

```python
class Pricer(Protocol):
    def supports(self, contract: Contract) -> bool
    def premium(self, contract) -> PriceResult          # value, stderr (0 for closed form)
    def price_along(self, paths, contract) -> np.ndarray     # (m, n+1), time-t money
    def greeks(self, paths, contract, which=("delta",)) -> dict[str, np.ndarray]
        # each (m, n+1, d): sensitivity of the contract's price to each instrument
```

`MonteCarloPricer(generator, n_inner, seed)` is the fallback: nested simulation from each
`(t_k, state)` with pathwise or bump deltas, cached per `(spec_id, split)`. It is slow by
construction; the validator warns when it is the only pricer for a classical hedger on
more than 5,000 paths.

## Instrument set

```python
@dataclass(frozen=True)
class InstrumentSet:
    # subset of generator instruments, plus derived
    names: tuple[str, ...]
    # e.g. VarianceSwapMark(tenor), ListedCall(strike, tenor)
    derived: tuple[DerivedInstrument, ...]
```

A derived instrument needs a pricer and adds a column to `instruments` before the
environment sees them.

## Features

```python
class FeatureBlock(Protocol):
    name: str
    out_dim: int
    requires: tuple[str, ...]        # state or feature names it reads
    def __call__(self, paths: TorchPaths, k: int, prev: torch.Tensor) -> torch.Tensor  # (m, out_dim)
```

Blocks: `Time`, `LogMoneyness(strike, scale="sqrt_tau")`, `Spot`, `Variance`,
`Greeks(which)`, `RealisedVol(window)`, `RegimeProb`, `PrevPosition`,
`Signature(depth, window)`, `Concat(blocks)`. The policy receives `Concat`'s output.

## Policy

```python
class Policy(nn.Module):
    n_in: int
    n_out: int                     # = number of instruments traded
    stateful: bool
    def initial_state(self, m: int) -> Any
    def forward(self, t: float, x: torch.Tensor, prev: torch.Tensor, state) -> tuple[torch.Tensor, Any]
        # x (m, n_in), prev (m, n_out) -> positions (m, n_out)

def rollout(policy, features: FeatureBlock, paths: TorchPaths) -> torch.Tensor   # (m, n, d)
```

One rollout for everything. Classical hedgers are `Policy` subclasses with no parameters
(`DeltaHedge`, `DeltaGammaHedge`, `DeltaVegaHedge`, `WhalleyWilmott`, `ConstantBand`,
`StaticReplication`); they read greeks from the pricer through a `Greeks` feature block.
Learned hedgers: `FeatureMLP(n_in, n_out, width, depth, activation)`,
`GRUPolicy(n_in, n_out, hidden)` (from PR 4), `PerStepMLP(n_steps, ...)` (the 2019
paper's separate weights per date), `OutputClip(lo, hi)` wrapper.

## Cost model

```python
class CostModel(Protocol):
    def amount(self, prices (m, n+1, d), positions (m, n, d), discount) -> np.ndarray      # (m,)
    def amount_torch(self, prices, positions, discount, *, exact: bool) -> torch.Tensor
    def label(self) -> str
    def params(self) -> dict
```

Promote `amount_torch`, `label`, `params` into the protocol (PR 2 found them by
`hasattr`). `discount` is passed so fixed cash costs are discounted correctly.

## Objective

```python
class Objective(Protocol):
    def __call__(self, pnl: np.ndarray) -> float
    def torch(self, pnl: torch.Tensor, aux: dict) -> torch.Tensor   # aux holds trainable thresholds
    def standard_error(self, pnl) -> float
    def params(self) -> dict
```

`Entropic(lam)`, `CVaR(alpha)`, `MeanVariance(lam)`, `Indifference(base: Objective)`.

## Environment and P&L

```python
@dataclass(frozen=True)
class Environment:
    generator: Generator
    instruments: InstrumentSet
    cost: CostModel
    def simulate(self, m, n, seed) -> Paths
    def pnl(self, paths, positions, contract, premium) -> np.ndarray
    def pnl_torch(self, paths, positions, contract, premium, *, exact) -> torch.Tensor
```

`pnl` and `pnl_torch` are the only two P&L functions in the package. The PR 1
`terminal_pnl` becomes `pnl`; the PR 2 `pnl_from_positions_torch` and the PR 4
`fit_policy` and PR 5 `ledger` accounting all collapse into `pnl_torch` and a
`ledger(...)` view that returns the per-step book from the same arithmetic.

## Graph spec

```json
{
  "schema": "deephedge.graph/v1",
  "id": "sha256 prefix of the canonical JSON without this field",
  "seeds": {"train": 0, "validation": 1, "test": 12345, "torch": 0},
  "nodes": {
    "mkt":  {"type": "generator.heston", "params": {"kappa": 1.0, "theta": 0.04, "xi": 2.0, "rho": -0.7, "v0": 0.04}},
    "con":  {"type": "contract.vanilla", "params": {"kind": "call", "strike": 100.0}},
    "ins":  {"type": "instruments", "params": {"names": ["spot", "variance_swap"]}},
    "cost": {"type": "cost.proportional", "params": {"rate": [0.005, 0.0]}},
    "feat": {"type": "features.concat", "params": {"blocks": ["time", "log_moneyness", "variance", "prev_position"]}},
    "h_delta": {"type": "hedger.delta_vega"},
    "h_net":  {"type": "hedger.feature_mlp", "params": {"width": 64, "depth": 3}},
    "obj":  {"type": "objective.cvar", "params": {"alpha": 0.5}},
    "eval": {"type": "evaluator.compare", "params": {"bootstrap": 2000, "report": ["mean", "cvar", "entropic", "turnover"]}}
  },
  "edges": [["mkt", "ins"], ["ins", "cost"], ["mkt", "feat"], ["feat", "h_net"], ["ins", "h_delta"],
            ["ins", "h_net"], ["con", "obj"], ["h_delta", "eval"], ["h_net", "eval"]],
  "run": {"m_train": 20000, "m_test": 20000, "n_steps": 30, "epochs": 100, "batch": 2000, "lr": 0.005}
}
```

The registry maps `type` strings to classes and their parameter schemas. `validate`
checks: every node type exists; parameter types and ranges; `feat.out_dim == h_net.n_in`;
`len(ins.names) == h.n_out` for every hedger; the contract's monitoring fits `n_steps`;
features requiring state names the generator provides; three distinct seeds; a pricer
exists for every classical hedger or the Monte Carlo fallback is explicitly enabled.
Cross-environment runs are a list of specs sharing hedger nodes with different `mkt`
nodes; the evaluator receives `(train_spec, test_spec)` pairs.
