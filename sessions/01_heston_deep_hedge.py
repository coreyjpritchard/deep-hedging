import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="01 Heston hedge")


@app.cell(hide_code=True)
def _():
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from deephedge.env import Environment
    from deephedge.policy import rollout_positions
    from deephedge.risk import summarise
    from deephedge.train import (
        DEFAULT_HESTON,
        cached_heston_policy,
        train_heston,
    )

    torch.set_num_threads(4)
    return (
        DEFAULT_HESTON,
        Environment,
        cached_heston_policy,
        go,
        mo,
        np,
        replace,
        rollout_positions,
        summarise,
        train_heston,
        torch,
    )


@app.cell(hide_code=True)
def _(DEFAULT_HESTON, mo):
    cost_bp = mo.ui.slider(
        steps=[0, 25, 50, 100],
        value=round(DEFAULT_HESTON.cost * 1e4),
        show_value=True,
        label="Proportional cost (basis points of the price traded)",
    )
    retrain = mo.ui.run_button(label="Retrain with this cost", kind="warn")
    mo.vstack(
        [
            mo.md(
                "# At zero cost the network and the model hedge land together\n"
                "A short 30-day at-the-money call in the Heston market of Buehler, "
                "Gonon, Teichmann and Wood (2019), hedged with the spot and a variance "
                "swap. The blue histogram is the model hedge, delta plus the swap weight "
                "of vega. The red one is a network trained to minimise 50% CVaR. "
                "The grey one uses only the spot. Move the cost."
            ),
            cost_bp,
            retrain,
        ]
    )
    return cost_bp, retrain


@app.cell(hide_code=True)
def _(DEFAULT_HESTON, cost_bp, replace):
    setup = replace(DEFAULT_HESTON, cost=cost_bp.value / 1e4)
    return (setup,)


@app.cell(hide_code=True)
def _(mo):
    get_trained, set_trained = mo.state({})
    return get_trained, set_trained


@app.cell(hide_code=True)
def _(mo, retrain, set_trained, setup, train_heston):
    retrain_note = mo.md("")
    if retrain.value:
        with mo.status.spinner(title="Training, about 30 s on four cores"):
            _result = train_heston(setup)
        set_trained(lambda d: {**d, setup.key(): _result.policy})
        retrain_note = mo.callout(
            mo.md(
                f"Trained for {setup.key()} in **{_result.wall_time:.1f} s**. "
                f"Training objective (CVaR at {setup.alpha}) fell from "
                f"{_result.history[0]:.3f} to {_result.history[-1]:.3f}."
            ),
            kind="success",
        )
    retrain_note
    return


@app.cell(hide_code=True)
def _(cached_heston_policy, get_trained, setup):
    _session = get_trained()
    if setup.key() in _session:
        policy, trained_for = _session[setup.key()], setup
        source = "retrained in this session"
    else:
        policy, trained_for = cached_heston_policy(setup)
        source = "cached weights"
    matches = trained_for.key() == setup.key()
    return matches, policy, source, trained_for


@app.cell(hide_code=True)
def _(DEFAULT_HESTON):
    # Paths and the model hedge do not depend on the cost slider.
    market = DEFAULT_HESTON.market()
    sim = market.simulate(DEFAULT_HESTON.m_eval, DEFAULT_HESTON.n_steps, seed=12345)
    benchmark = market.benchmark_positions(sim)
    return benchmark, market, sim


@app.cell(hide_code=True)
def _(policy, rollout_positions, sim, torch):
    with torch.no_grad():
        network_pos = rollout_positions(policy, sim.features, sim.time, 2).double().numpy()
    return (network_pos,)


@app.cell(hide_code=True)
def _(Environment, benchmark, market, network_pos, setup, sim, summarise):
    env = Environment(market, setup.cost)
    spot_only = benchmark.copy()
    spot_only[:, :, 1] = 0.0
    pnl = {
        "Model hedge": env.pnl(sim, benchmark),
        "Network": env.pnl(sim, network_pos),
        "Spot only": env.pnl(sim, spot_only),
    }
    stats = {name: summarise(x, setup.alpha, 1.0) for name, x in pnl.items()}
    return pnl, stats


@app.cell(hide_code=True)
def _(go, np, pnl, setup, stats):
    colours = {"Model hedge": "#1f77b4", "Network": "#d62728", "Spot only": "#7f7f7f"}
    _all = np.concatenate(list(pnl.values()))
    _edges = np.linspace(np.quantile(_all, 0.005), np.quantile(_all, 0.995), 81)
    _centres = 0.5 * (_edges[1:] + _edges[:-1])
    fig_pnl = go.Figure()
    for _name, _x in pnl.items():
        _density, _ = np.histogram(_x, bins=_edges, density=True)
        _s = stats[_name]
        fig_pnl.add_bar(
            x=_centres,
            y=_density,
            width=_edges[1] - _edges[0],
            name=(f"{_name}: mean {_s['mean']:+.2f}, std {_s['std']:.2f}, CVaR {_s['cvar']:.2f}"),
            marker_color=colours[_name],
            opacity=0.55,
        )
    fig_pnl.update_layout(
        barmode="overlay",
        title=(
            f"Terminal P&L of the short call, {setup.m_eval:,} paths, "
            f"cost {setup.cost * 1e4:.0f} bp. CVaR is the worst half of losses."
        ),
        xaxis_title="P&L (premium - payoff + trading gains - costs)",
        yaxis_title="density",
        legend={"orientation": "h", "y": -0.25},
        height=460,
        margin={"t": 60, "b": 10},
    )
    return (fig_pnl,)


@app.cell(hide_code=True)
def _(fig_pnl, matches, mo, setup, source, trained_for):
    _note = (
        mo.md(f"Network: {source}, trained for exactly this cost.")
        if matches
        else mo.callout(
            mo.md(
                f"No cached network for {setup.key()}. Showing the network trained for "
                f"{trained_for.key()}, evaluated at this cost. Press "
                "*Retrain with this cost* to train one (about 30 s)."
            ),
            kind="warn",
        )
    )
    mo.vstack([_note, fig_pnl])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the histogram is

    The hedger sells the call at its Heston price and trades two instruments,
    the spot and a variance swap of strike 0. The swap pays the integrated
    variance over the month, not that integral divided by $T$. With
    $v_0 = \theta$ and $r = 0$ it is worth $\theta T$ at inception, about
    0.0033. The P&L is the premium, minus the call payoff, plus gains on
    both instruments, minus proportional costs on both. Position $k$ is
    held from date $k$ to the next date. The book is opened at $t = 0$ and
    liquidated at maturity, and both of those trades are charged.

    The model hedge holds the call's delta in the spot and
    $\partial C / \partial v$ divided by $\partial P / \partial v$ in the
    swap, where $P$ is the swap's price. That is equation (5.6) of Buehler,
    Gonon, Teichmann and Wood (2019, *Deep hedging*, Quantitative Finance
    19(8)). The grey histogram is the same delta with the swap weight set
    to zero.

    ## The figure in the paper, and this one

    Their Figure 2 is this histogram at zero transaction cost, on one
    million paths. Parameters, section 5.2: $T = 30/365$, daily steps,
    $S_0 = K = 100$, $\kappa = 1$, $\theta = v_0 = 0.04$, $\xi = 2$,
    $\rho = -0.7$, and no drift in the spot. Feller's condition fails:
    $2\kappa\theta = 0.08 < \xi^2 = 4$. Both books are sold at the
    risk-neutral price, which they print as 1.69. The model hedge and the
    network lie on top of each other. The indifference price at
    $\alpha = 0.5$, the cash that makes the hedged book acceptable under
    their risk measure, is 1.94. Their variance is an exact CIR draw and
    the spot is a simplified Broadie-Kaya step. Their network has its own
    weights on each of the 30 dates.

    This picture is the same comparison on 20 000 paths, seed 12345. The
    premium is the quadrature price 1.692, their 1.69 to the digit they
    print. The variance is a full truncation Euler step, the scheme used
    for rough volatility in the sibling `roughvol` package, and the network
    is one set of weights shared across dates. Its inputs are time to
    maturity, log moneyness, variance over $\theta$, and the position it
    already holds in each instrument. It minimises 50% CVaR, the mean loss
    on the worst half of paths, which is their average value at risk at
    $\alpha = 0.5$.

    At zero cost, on this sample, the model hedge has mean $-0.03$,
    standard deviation $0.40$ and CVaR $0.30$. The network has $-0.01$,
    $0.44$ and $0.31$. Premium plus CVaR is an indifference price of
    $1.99$ for the model hedge and $2.00$ for the network, against their
    $1.94$. The histograms overlap, which is what their figure shows. The
    network does not beat the model hedge: $2.00$ against $1.99$. A
    bootstrap standard error of each CVaR is about $0.004$, so the $0.05$
    gap against $1.94$ is not the noise of this sample.

    The gap is the coarse variance step. Simulate the same seed on 480
    steps and keep every 16th date, so the hedge still trades on the same
    30 dates, and the model hedge's premium plus CVaR is $1.934$. On 1920
    steps it is $1.936$. The network trained on 30-step paths scores
    $1.971$ on the 480-step paths, so there it trails the model hedge by
    about $0.04$. On the 30-step grid full truncation holds the variance
    at zero on 45 per cent of hedging dates, and the variance swap's mark
    is not a discrete martingale: its mean at maturity exceeds its price
    at inception by $2.4 \times 10^{-4}$, about 7 per cent of the mark and
    6.5 standard errors. On 480 steps that drift is within noise, and 480
    steps is the grid on which the martingale test in
    `tests/test_heston.py` runs. These sub-stepped numbers come from a
    one-off check outside this session; the session itself simulates on
    the 30 hedging dates.

    The grey histogram is the control. Delta alone, no variance swap, has
    standard deviation $1.56$ and CVaR $1.28$. The tight blue and red
    histograms are the swap, not a cleverer spot hedge. No hedge at all
    has mean $-0.11$, standard deviation $2.27$ and CVaR $1.70$.

    ## What the cost slider does

    At 100 bp the model hedge still holds the zero-cost greek weights, so
    it pays to trade. Its mean P&L is $-2.80$ and its CVaR is $3.52$. The
    network trained at 100 bp holds less of both instruments. Its mean is
    $-0.42$, its standard deviation $1.77$ and its CVaR $1.54$.

    Most of that cost is the spot. The swap's mark is about $0.003$, so a
    proportional charge on the mark, even on a position of a few hundred
    swaps, is a few hundredths. Half a share of a spot near 100, at 100 bp,
    costs about $0.5$ to open, before any rebalancing. A cost that should
    matter for the variance swap has to be levied on a notional that is not
    this mark. That belongs on the cost model, not inside the simulator.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                "## Try this\nWrite down a guess before you move the slider. Then open the answer."
            ),
            mo.accordion(
                {
                    "1. Set the cost to 0. Do the red and blue histograms separate, "
                    "and what is the grey one saying?": mo.md(
                        "They do not separate in any way that matters: CVaR 0.31 for the "
                        "network against 0.30 for the model hedge, standard deviations 0.44 "
                        "and 0.40. The grey histogram is delta with no variance swap. Its "
                        "CVaR is 1.28 and its standard deviation 1.56. The tight hedge is "
                        "the swap."
                    ),
                    "2. Set the cost to 100 bp. Which book loses more?": mo.md(
                        "The model hedge. Its mean P&L is -2.80 and its CVaR is 3.52, because "
                        "it keeps trading the zero-cost weights. The network trained at 100 bp "
                        "has mean -0.42 and CVaR 1.54. If the note above the figure says the "
                        "weights were trained at 0 bp, you are looking at the zero-cost network "
                        "paying this cost, which behaves like the model hedge. Press retrain."
                    ),
                    "3. The paper's indifference price at alpha 0.5 is 1.94. What is ours?": mo.md(
                        "1.99 for the model hedge on this sample: the quadrature premium 1.692 "
                        "plus a CVaR of 0.30. The network is 2.00. Their printed risk-neutral "
                        "price is 1.69, and 1.692 rounds to that. The indifference prices "
                        "differ by about 0.05. The CVaR here has a bootstrap standard error "
                        "of about 0.004, so the gap is not noise. It is the coarse variance "
                        "step: simulated on 480 steps and hedged on the same 30 dates, the "
                        "model hedge gives 1.934."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
