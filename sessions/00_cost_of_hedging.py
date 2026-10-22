import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="00 Cost of hedging")


@app.cell(hide_code=True)
def _():
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from deephedge.policy import held_positions, position_grid, rollout_paths
    from deephedge.risk import summarise
    from deephedge.train import DEFAULT, cached_policy, train_setup

    torch.set_num_threads(4)
    return (
        DEFAULT,
        cached_policy,
        go,
        mo,
        held_positions,
        np,
        position_grid,
        replace,
        rollout_paths,
        summarise,
        train_setup,
    )


@app.cell(hide_code=True)
def _(DEFAULT, mo):
    cost_bp = mo.ui.slider(
        0, 200, step=25, value=round(DEFAULT.cost * 1e4), show_value=True,
        label="Proportional cost (basis points of notional traded)",
    )
    n_steps = mo.ui.slider(
        steps=[10, 30, 60, 120], value=DEFAULT.n_steps, show_value=True,
        label="Rehedging steps over 30 days",
    )
    lam = mo.ui.slider(
        steps=[0.1, 0.3, 1.0, 3.0, 10.0], value=DEFAULT.lam, show_value=True,
        label="Risk aversion: entropic lambda",
    )
    retrain = mo.ui.run_button(label="Retrain with these settings", kind="warn")
    mo.vstack([
        mo.md(
            "# Delta hedging loses money when trading costs money\n"
            "A short 30-day at-the-money call (S0 = K = 100, sigma = 20%), hedged three "
            "ways: Black-Scholes delta, a small neural network trained to minimise the "
            "entropic risk of the hedged P&L, and no hedge at all. Move the sliders."
        ),
        cost_bp,
        n_steps,
        lam,
        retrain,
    ])
    return cost_bp, lam, n_steps, retrain


@app.cell(hide_code=True)
def _(DEFAULT, cost_bp, lam, n_steps, replace):
    setup = replace(DEFAULT, cost=cost_bp.value / 1e4, n_steps=n_steps.value, lam=lam.value)
    return (setup,)


@app.cell(hide_code=True)
def _(mo):
    get_trained, set_trained = mo.state({})
    return get_trained, set_trained


@app.cell(hide_code=True)
def _(mo, retrain, set_trained, setup, train_setup):
    retrain_note = mo.md("")
    if retrain.value:
        _estimate = round(0.3 * setup.epochs * setup.n_steps / 30)
        with mo.status.spinner(title=f"Training, roughly {_estimate} s on four cores"):
            _result = train_setup(setup)
        set_trained(lambda d: {**d, setup.key(): _result.policy})
        retrain_note = mo.callout(
            mo.md(
                f"Trained for {setup.key()} in **{_result.wall_time:.1f} s**. "
                f"Training objective (entropic risk of P&L) fell from {_result.history[0]:.3f} "
                f"to {_result.history[-1]:.3f}."
            ),
            kind="success",
        )
    retrain_note
    return


@app.cell(hide_code=True)
def _(cached_policy, get_trained, setup):
    _session = get_trained()
    if setup.key() in _session:
        policy, trained_for = _session[setup.key()], setup
        source = "retrained in this session"
    else:
        policy, trained_for = cached_policy(setup)
        source = "cached weights"
    matches = trained_for.key() == setup.key()
    return matches, policy, source, trained_for


@app.cell(hide_code=True)
def _(
    np,
    policy,
    rollout_paths,
    setup,
    summarise,
):
    env = setup.environment()
    sim = env.market.simulate(setup.m_eval, setup.n_steps, seed=12345)
    eval_paths = sim.spot
    pnl = {
        "Delta hedge": env.pnl(sim, env.market.benchmark_positions(sim)),
        "Network": env.pnl(sim, rollout_paths(policy, sim)),
        "No hedge": env.pnl(sim, np.zeros((setup.m_eval, setup.n_steps))),
    }
    stats = {name: summarise(x, setup.alpha, setup.lam) for name, x in pnl.items()}
    return eval_paths, pnl, stats


@app.cell(hide_code=True)
def _(go, np, pnl, setup, stats):
    colours = {"Delta hedge": "#1f77b4", "Network": "#d62728", "No hedge": "#7f7f7f"}
    _all = np.concatenate(list(pnl.values()))
    _edges = np.linspace(np.quantile(_all, 0.003), np.quantile(_all, 0.999), 121)
    _centres = 0.5 * (_edges[1:] + _edges[:-1])
    fig_pnl = go.Figure()
    for _name, _x in pnl.items():
        _density, _ = np.histogram(_x, bins=_edges, density=True)
        _s = stats[_name]
        fig_pnl.add_bar(
            x=_centres,
            y=_density,
            width=_edges[1] - _edges[0],
            name=(
                f"{_name}: mean {_s['mean']:+.2f}, std {_s['std']:.2f}, "
                f"CVaR {_s['cvar']:.2f}, entropic {_s['entropic']:.2f}"
            ),
            marker_color=colours[_name],
            opacity=0.55,
        )
    fig_pnl.update_layout(
        barmode="overlay",
        title=(
            f"(a) Terminal P&L of the short call, {setup.m_eval:,} fresh paths. "
            f"CVaR at {setup.alpha}, entropic at lambda {setup.lam:g}"
        ),
        xaxis_title="P&L (premium - payoff + trading gains - costs)",
        yaxis_title="density",
        legend={"orientation": "h", "y": -0.25},
        height=430,
        margin={"t": 60, "b": 10},
    )
    return colours, fig_pnl


@app.cell(hide_code=True)
def _(
    colours,
    eval_paths,
    go,
    held_positions,
    np,
    policy,
    position_grid,
    setup,
    trained_for,
):
    _k = setup.n_steps // 2
    _t = _k * setup.maturity / setup.n_steps
    _spot, _held = held_positions(policy, eval_paths, setup.maturity, _k)
    _show = np.random.default_rng(0).choice(_spot.size, 3_000, replace=False)
    _spots = np.linspace(88, 112, 241)
    _shown = np.array([0.2, 0.5, 0.8])
    _positions = position_grid(policy, _t, _spots, _shown)
    _delta = setup.market().delta(_spots, setup.maturity - _t)

    fig_band = go.Figure()
    fig_band.add_scattergl(
        x=_spot[_show], y=_held[_show], mode="markers",
        marker={"color": colours["Network"], "size": 3, "opacity": 0.25},
        name="position the network holds at T/2, one dot per path",
        hoverinfo="skip",
    )
    fig_band.add_scatter(
        x=_spots, y=_delta, name="Black-Scholes delta",
        line={"color": colours["Delta hedge"], "width": 3},
    )
    for _p, _y, _dash in zip(_shown, _positions, ["dot", "dash", "dashdot"], strict=True):
        fig_band.add_scatter(
            x=_spots, y=_y, name=f"network's new position if it holds {_p:.1f}",
            line={"color": "#2b2b2b", "dash": _dash, "width": 1.8},
        )
    fig_band.update_layout(
        title=(
            f"(b) Positions at t = T/2, network trained at {trained_for.cost * 1e4:.0f} bp, "
            f"{trained_for.n_steps} steps, lambda {trained_for.lam:g}"
        ),
        xaxis_title="spot S",
        xaxis_range=[88, 112],
        yaxis_title="position in the underlying",
        yaxis_range=[-0.05, 1.05],
        legend={"orientation": "h", "y": -0.25},
        height=500,
        margin={"t": 60, "b": 10},
    )
    return (fig_band,)


@app.cell(hide_code=True)
def _(fig_band, fig_pnl, matches, mo, setup, source, trained_for):
    _note = (
        mo.md(f"Network: {source}, trained for exactly these settings.")
        if matches
        else mo.callout(
            mo.md(
                f"No cached network for {setup.key()}. Showing the network trained for "
                f"{trained_for.key()}, evaluated under the current settings. Press "
                "*Retrain with these settings* to train one (about 30 s at 30 steps)."
            ),
            kind="warn",
        )
    )
    mo.vstack([_note, fig_pnl, fig_band])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the two panels show

    Panel (a) is the distribution of the hedger's final P&L. The hedger sells the call at
    its Black-Scholes price, trades the stock, pays $c\,S\,|\Delta \text{position}|$ on
    every trade (including opening and closing the hedge), and pays the payoff at expiry.
    With no costs the delta hedge is centred on zero and narrow. Turn costs on and it
    slides left: delta hedging pays to follow every wiggle of the stock.

    Panel (b) shows where the network's positions sit halfway to expiry. Each red dot is
    one simulated path: the spot at $t = T/2$ and the position the network holds after
    trading there. With no costs the dots lie on the blue delta curve. With costs they
    spread into a band around it: the network lets its position drift away from delta and
    trades only when the gap gets large. The three black curves are the decision rule
    itself. Fix the position the network holds coming in and read off the position it
    chooses as the spot varies. Where a curve is flatter than delta the network trades
    less than delta would; at zero cost all three lie on delta.

    ## Why delta hedging bleeds under costs

    The cost of delta hedging is proportional to the total variation of delta, and that
    grows like $\sqrt{n}$. Between two rebalancing dates a time $\Delta t$ apart, delta
    moves by about $\Gamma S \sigma \sqrt{\Delta t}\,Z$ with $Z$ standard normal. Each
    rebalance therefore costs about $c\,\sigma\,\Gamma S^2 \sqrt{\Delta t}\,\sqrt{2/\pi}$
    in expectation, and summing over $n = T/\Delta t$ steps gives

    $$
    \mathbb{E}[\text{cost}] \approx c\,\sigma\sqrt{\tfrac{2}{\pi}}\;\sqrt{\tfrac{n}{T}}
    \int_0^T \mathbb{E}\big[\Gamma_t S_t^2\big]\,dt .
    $$

    Meanwhile the hedging error without costs has a standard deviation that falls like
    $1/\sqrt{n}$. Rehedging more often trades one problem for the other, and no choice
    of $n$ makes both small. Leland (1985, *Journal of Finance* 40(5)) used the same
    computation to price options with an adjusted volatility.

    ## What the network learns

    The network learns to hedge less: it holds a band around delta and trades when the
    spot pushes its position too far from it. Whalley and Wilmott (1997, *An asymptotic
    analysis of an optimal hedging model for option pricing with transaction costs*,
    Mathematical Finance 7(3)) derive the optimal band for small costs under exponential
    utility with risk aversion $\gamma$. Inside it you do nothing; outside it you trade to
    the nearest edge. Its half-width around delta is

    $$
    \left( \frac{3}{2}\, \frac{c\, S\, e^{-r(T-t)}\, \Gamma^2}{\gamma} \right)^{1/3},
    $$

    so it is widest where gamma is large (near the money, near expiry) and grows like the
    cube root of the cost. Nobody told the network this. Its inputs are time to maturity,
    log moneyness and the position it already holds, as in the feature-based network of
    Buehler, Gonon, Teichmann and Wood (2019, *Deep hedging*, Quantitative Finance
    19(8)). The learned band is softer than the theoretical one: the black curves are
    flatter than delta but not flat, so inside the band the network still trades a
    little. Session 03 puts the formula and the learned band on the same axes and asks
    whether a sharp band would do better.

    A band policy has a second advantage that panel (a) shows when you change the number
    of steps: its cost grows far more slowly than $\sqrt{n}$, because inside the band it
    does little however often it is allowed to look.

    ## What is being minimised

    The objective is a convex risk measure of the P&L, not its variance. The network
    minimises the entropic risk

    $$
    \rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\big[e^{-\lambda X}\big],
    $$

    estimated on 20 000 simulated paths, with the gradient taken through every trade and
    every cost of the simulation. Variance would punish a large gain as much as a large
    loss; the entropic risk weighs losses exponentially and gains hardly at all. As
    $\lambda \to 0$ it tends to $-\mathbb{E}[X]$, a risk-neutral hedger who only minimises
    expected cost; large $\lambda$ cares mostly about the worst paths. The measure is
    monetary, $\rho(X + m) = \rho(X) - m$, so $\rho$ of the hedged P&L is the cash the
    seller needs on top of the Black-Scholes premium to make the trade acceptable.
    Buehler et al. use this to define an indifference price.

    Minimising $\rho_\lambda$ is the same as maximising expected exponential utility,
    which is the Whalley and Wilmott criterion with $\gamma = \lambda$. It is also time
    consistent: the best trade today does not depend on the P&L made so far, so a network
    that sees only time, spot and its position can represent the optimum. CVaR, the other
    measure in Buehler et al. and printed in panel (a), lacks that property. Without
    costs and with frequent rehedging the market is complete and every sensible objective
    leads back to delta. Costs make the market incomplete, and then the risk measure
    shapes the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(
            "## Try this\n"
            "Write down a guess before you move the slider. Then open the answer."
        ),
        mo.accordion({
            "1. Set the cost to 0. Does the network beat delta hedging? What happens "
            "to the band in panel (b)?": mo.md(
                "No. With no costs the network has nothing to gain from inaction, so it "
                "learns delta: the two histograms coincide (entropic risk 0.062 against "
                "0.063 on the default sample) and the dots and black curves in panel "
                "(b) fall onto the blue line."
            ),
            "2. Set the cost to 100 bp and move the steps from 10 to 120. By how much "
            "does delta hedging's mean P&L fall? And the network's?": mo.md(
                "Delta hedging's mean P&L goes from -1.96 at 10 steps to -4.48 at 120. "
                "The rebalancing cost grows like the square root of $n$, so four times "
                "the steps costs about twice as much. The network, trained at 30 steps "
                "and evaluated at the others, goes from -1.26 to -1.89, and its "
                "standard deviation falls: a band policy does nothing inside the band "
                "however often it is allowed to look. Retrain at 120 steps and compare."
            ),
            "3. Go from 50 bp to 100 bp. How much wider does the band get at the "
            "money?": mo.md(
                "Whalley and Wilmott predict a factor $2^{1/3} \\approx 1.26$, not 2. "
                "The network's band barely widens: near S = 100 the spread of its "
                "positions around delta grows by about 5 per cent, and the band moves "
                "upwards instead, so the network holds more stock than delta. Small "
                "costs, discrete rehedging and the network's soft band are three "
                "candidate reasons. Session 03 separates them."
            ),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why this matters for the capstone

    Here the network only needs $(t, S_t)$ and its previous position because the
    Black-Scholes market is Markov in the spot: everything the future depends on is in
    today's price. Under rough volatility that fails. The variance in rough Bergomi is
    driven by a fractional kernel with Hurst exponent near 0.1, so the law of the future
    depends on the whole past path of the volatility driver, and the optimal hedge ratio
    is not a function of $(t, S)$ alone. Horvath, Teichmann and Zuric (2021, *Deep
    hedging under rough volatility*, Risks 9(7)) test the original feature-based
    architecture on rough Bergomi paths and propose networks that carry a memory of the
    path. Sessions 03 and 04 of this project build up to that setting, using the rough
    Bergomi simulator from the sibling `rough-volatility` repository.
    """)
    return


if __name__ == "__main__":
    app.run()
