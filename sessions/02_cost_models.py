import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="02 Cost models")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from deephedge.env import Environment
    from deephedge.policy import (
        held_instrument,
        held_positions,
        heston_spot_response,
        position_grid,
        rollout_paths,
        rollout_positions,
    )
    from deephedge.risk import summarise
    from deephedge.train import (
        COST_SESSION,
        cost_model_from_spec,
        cost_session_market,
        load_cost_session_policy,
        train_cost_session,
    )

    torch.set_num_threads(4)
    return (
        COST_SESSION,
        Environment,
        cost_model_from_spec,
        cost_session_market,
        go,
        held_instrument,
        held_positions,
        heston_spot_response,
        load_cost_session_policy,
        mo,
        np,
        position_grid,
        rollout_paths,
        rollout_positions,
        summarise,
        torch,
        train_cost_session,
    )


@app.cell(hide_code=True)
def _(mo):
    market = mo.ui.radio(
        options={"Black-Scholes": "bs", "Heston": "heston"},
        value="Black-Scholes",
        inline=True,
        label="Market",
    )
    mo.vstack([
        mo.md(
            "# The cost model changes what the network holds\n"
            "A short 30-day at-the-money call. Pick the market and the cost. "
            "The histogram is the hedged P&L. The second panel is the position "
            "the network learned for that cost."
        ),
        market,
    ])
    return (market,)


@app.cell(hide_code=True)
def _(COST_SESSION, market, mo):
    _options = {
        spec["label"]: key
        for key, spec in COST_SESSION.items()
        if spec["market"] == market.value
    }
    _default = "bs-prop" if market.value == "bs" else "heston-mark"
    _name = next(label for label, key in _options.items() if key == _default)
    cost = mo.ui.radio(options=_options, value=_name, label="Cost model")
    cost
    return (cost,)


@app.cell(hide_code=True)
def _(mo):
    scale = mo.ui.slider(
        steps=[0, 0.5, 1, 1.5, 2],
        value=1,
        show_value=True,
        label="Charge, as a multiple of this model's rates (1 is the training cost)",
    )
    retrain = mo.ui.run_button(label="Retrain this cost model", kind="warn")
    mo.vstack([scale, retrain])
    return retrain, scale


@app.cell(hide_code=True)
def _(mo):
    get_trained, set_trained = mo.state({})
    return get_trained, set_trained


@app.cell(hide_code=True)
def _(cost):
    key = cost.value
    return (key,)


@app.cell(hide_code=True)
def _(
    COST_SESSION,
    key,
    mo,
    retrain,
    set_trained,
    train_cost_session,
):
    retrain_note = mo.md("")
    if retrain.value:
        with mo.status.spinner(title="Training on four cores"):
            _result = train_cost_session(key)
        set_trained(lambda d: {**d, key: _result.policy})
        _spec = COST_SESSION[key]
        retrain_note = mo.callout(
            mo.md(
                f"Trained {key} in **{_result.wall_time:.1f} s**. "
                f"The exact objective on the training paths went from "
                f"{_result.history[0]:.3f} to {_result.history[-1]:.3f}. "
                f"Sample size {_spec['m_train']:,}, {_spec['epochs']} epochs."
            ),
            kind="success",
        )
    retrain_note
    return


@app.cell(hide_code=True)
def _(get_trained, key, load_cost_session_policy):
    _session = get_trained()
    if key in _session:
        policy = _session[key]
        source = "retrained in this session"
    else:
        policy, _spec = load_cost_session_policy(key)
        source = "cached weights"
    return policy, source


@app.cell(hide_code=True)
def _(COST_SESSION, cost_model_from_spec, key, scale):
    spec = COST_SESSION[key]
    model = cost_model_from_spec(spec, float(scale.value))
    return model, spec


@app.cell(hide_code=True)
def _(cost_session_market, market):
    name = market.value
    mkt = cost_session_market(name)
    n_paths = 20_000 if name == "bs" else 4_000
    sim = mkt.simulate(n_paths, 30, seed=12345)
    bench = mkt.benchmark_positions(sim)
    return bench, mkt, n_paths, sim


@app.cell(hide_code=True)
def _(market, policy, rollout_paths, rollout_positions, sim, torch):
    if market.value == "bs":
        net = rollout_paths(policy, sim)
    else:
        with torch.no_grad():
            net = rollout_positions(policy, sim.features, sim.time, 2).double().numpy()
    return (net,)


@app.cell(hide_code=True)
def _(Environment, bench, mkt, model, net, np, sim, spec, summarise):
    env = Environment(mkt, model)
    flat = np.zeros_like(bench)
    if bench.shape[-1] == 1:
        books = {
            "Delta hedge": bench,
            "Network": net,
            "No hedge": flat,
        }
    else:
        spot_only = bench.copy()
        spot_only[:, :, 1] = 0.0
        books = {
            "Model hedge": bench,
            "Network": net,
            "Spot only": spot_only,
        }
    pnl = {name: env.pnl(sim, pos) for name, pos in books.items()}
    stats = {
        name: summarise(x, spec["alpha"], spec["lam"]) for name, x in pnl.items()
    }
    return pnl, stats


@app.cell(hide_code=True)
def _(go, np, pnl, scale, spec, stats):
    colours = {
        "Delta hedge": "#1f77b4",
        "Model hedge": "#1f77b4",
        "Network": "#d62728",
        "No hedge": "#7f7f7f",
        "Spot only": "#7f7f7f",
    }
    _all = np.concatenate(list(pnl.values()))
    _edges = np.linspace(np.quantile(_all, 0.005), np.quantile(_all, 0.995), 81)
    _centres = 0.5 * (_edges[1:] + _edges[:-1])
    fig_pnl = go.Figure()
    for _name, _x in pnl.items():
        _density, _ = np.histogram(_x, bins=_edges, density=True)
        _s = stats[_name]
        if spec["market"] == "bs":
            _legend = (
                f"{_name}: mean {_s['mean']:+.2f}, std {_s['std']:.2f}, "
                f"CVaR {_s['cvar']:.2f}, entropic {_s['entropic']:.2f}"
            )
        else:
            _legend = (
                f"{_name}: mean {_s['mean']:+.2f}, std {_s['std']:.2f}, "
                f"CVaR {_s['cvar']:.2f}"
            )
        fig_pnl.add_bar(
            x=_centres,
            y=_density,
            width=_edges[1] - _edges[0],
            name=_legend,
            marker_color=colours[_name],
            opacity=0.55,
        )
    if spec["market"] == "bs":
        _level = "CVaR at 0.95 and entropic risk at lambda 1"
    else:
        _level = "CVaR at 0.5"
    fig_pnl.update_layout(
        barmode="overlay",
        title=(
            f"(a) Terminal P&L, {spec['label']}, "
            f"charge x{scale.value:g}. {_level}."
        ),
        xaxis_title="P&L (premium - payoff + gains - costs)",
        yaxis_title="density",
        legend={"orientation": "h", "y": -0.25},
        height=430,
        margin={"t": 60, "b": 10},
    )
    return colours, fig_pnl


@app.cell(hide_code=True)
def _(
    bench,
    colours,
    go,
    held_instrument,
    held_positions,
    heston_spot_response,
    mkt,
    net,
    np,
    policy,
    position_grid,
    sim,
    spec,
):
    _k = sim.n_steps // 2
    _t = float(sim.time[_k])
    fig_pos = go.Figure()
    if spec["market"] == "bs":
        _spot, _held = held_positions(policy, sim.spot, mkt.maturity, _k)
        _show = np.random.default_rng(0).choice(_spot.size, 3_000, replace=False)
        _spots = np.linspace(88, 112, 161)
        _prevs = np.array([0.2, 0.5, 0.8])
        _grid = position_grid(policy, _t, _spots, _prevs)
        _delta = mkt.delta(_spots, mkt.maturity - _t)
        fig_pos.add_scattergl(
            x=_spot[_show],
            y=_held[_show],
            mode="markers",
            marker={"color": colours["Network"], "size": 3, "opacity": 0.25},
            name="position held at T/2, one dot per path",
            hoverinfo="skip",
        )
        fig_pos.add_scatter(
            x=_spots,
            y=_delta,
            name="Black-Scholes delta",
            line={"color": colours["Delta hedge"], "width": 3},
        )
        for _p, _y, _dash in zip(_prevs, _grid, ["dot", "dash", "dashdot"], strict=True):
            fig_pos.add_scatter(
                x=_spots,
                y=_y,
                name=f"new position if it already holds {_p:.1f}",
                line={"color": "#2b2b2b", "dash": _dash, "width": 1.8},
            )
        fig_pos.update_layout(
            title="(b) Spot position at t = T/2. Black curves: the decision rule.",
            xaxis_title="spot S",
            xaxis_range=[88, 112],
            yaxis_title="position in the spot",
            yaxis_range=[-0.05, 1.05],
        )
    else:
        _spot, _swap = held_instrument(
            policy, sim.features, sim.time, 2, _k, instrument=1
        )
        _n_show = min(3_000, int(_spot.size))
        _show = np.random.default_rng(0).choice(_spot.size, size=_n_show, replace=False)
        _spots = np.linspace(80, 120, 81)
        _rule = heston_spot_response(policy, _t, _spots, mkt.theta, 0.5, 0.0)[:, 1]
        fig_pos.add_scattergl(
            x=sim.spot[_show, _k],
            y=bench[_show, _k, 1],
            mode="markers",
            marker={"color": colours["Model hedge"], "size": 3, "opacity": 0.2},
            name="model hedge, variance-swap weight at T/2",
            hoverinfo="skip",
        )
        fig_pos.add_scattergl(
            x=_spot[_show],
            y=_swap[_show],
            mode="markers",
            marker={"color": colours["Network"], "size": 3, "opacity": 0.35},
            name="network, variance-swap weight at T/2",
            hoverinfo="skip",
        )
        fig_pos.add_scatter(
            x=_spots,
            y=_rule,
            name="network's swap weight if it holds 0.5 shares and no swap",
            line={"color": "#2b2b2b", "width": 2},
        )
        _ys = np.concatenate([bench[:, _k, 1], _swap])
        _lo, _hi = np.quantile(_ys, [0.02, 0.98])
        _pad = 0.08 * (_hi - _lo + 1.0)
        fig_pos.update_layout(
            title="(b) Variance-swap position at t = T/2.",
            xaxis_title="spot S",
            xaxis_range=[80, 120],
            yaxis_title="position in the variance swap",
            yaxis_range=[float(_lo - _pad), float(_hi + _pad)],
        )
    fig_pos.update_layout(
        legend={"orientation": "h", "y": -0.25},
        height=500,
        margin={"t": 60, "b": 10},
    )
    return (fig_pos,)


@app.cell(hide_code=True)
def _(fig_pnl, fig_pos, mo, n_paths, scale, source, spec):
    if scale.value == 1:
        _charge = "The histogram uses the training cost."
    else:
        _charge = (
            f"The histogram uses {scale.value:g} times the training cost. "
            "The position panel does not: it is the network trained at scale 1."
        )
    mo.vstack([
        mo.md(
            f"Network: {source}, {spec['label']}. "
            f"{n_paths:,} paths, seed 12345. {_charge}"
        ),
        fig_pnl,
        fig_pos,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the two panels show

    Panel (a) is the terminal P&L of a short call sold at the model premium.
    The book is flat before the first date and flat again at maturity, and
    both of those trades are charged. Liquidation is not optional.

    Panel (b) is the position halfway to expiry. On Black-Scholes each red
    dot is one path, and the black curves are the decision rule for three
    positions the network might already hold. Where those curves separate,
    the network's next trade depends on what it already holds: that is the
    no-trade band. On Heston the dots are the variance-swap weight, model
    hedge in blue and network in red.

    The charge slider rescores the same positions. It does not change the
    network. Move it to see the histogram shift while panel (b) stays put.

    ## Four costs, on 20 000 Black-Scholes paths

    Premium 2.29. Seed 12345. Entropic risk at lambda 1, which is the
    objective the Black-Scholes networks minimise. Standard errors of the
    means below are under 0.01.

    With no cost the network and delta sit together: mean about 0.00,
    standard deviation 0.36, entropic risk 0.06. At S = 100 the decision
    rule ignores the incoming position. The three black curves lie on delta.

    With a 50 bp proportional cost the network's mean is -0.83 and its
    entropic risk is 1.05. Delta's are -1.37 and 1.54. At S = 100, moving
    the incoming position from 0.2 to 0.8 moves the chosen position by 0.66
    of that gap. The network lets the hedge drift, then trades.

    With a quadratic impact of 0.02 the network's mean is -0.54 and its
    entropic risk is 0.94. Delta's are -1.88 and 2.46. Opening half a share
    costs impact times (1/2 squared) times the spot, about 0.50 before any
    rebalancing, so delta's large trades are expensive. The band is still
    there: the same 0.2 to 0.8 move in the incoming position changes the
    chosen position by 0.57 of the gap.

    With a fixed charge of 0.05 per instrument per date on which the
    position changes, the network trades on 28 per cent of dates and delta
    on 93 per cent. Means are -1.50 and -1.55. Entropic risk is 1.79 for
    the network and 1.61 for delta, so on the objective it was trained for
    the network does not beat delta. At S = 100 the three incoming
    positions all map to about 0.55: there is no band. A fixed charge does
    not shrink when the trade shrinks, so the soft band that helps under a
    proportional cost has nothing to sell. The training surrogate uses
    width 0.05. The plotted P&L uses the exact indicator.

    ## The variance swap, on 4 000 Heston paths

    Same contract, Heston parameters of Buehler, Gonon, Teichmann and Wood
    (2019), section 5.2, under the pricing measure. Premium 1.692. The
    network minimises 50 per cent CVaR. It is smaller than the session 01
    network: width 32, depth 2, 8 000 training paths, 50 epochs. Standard
    errors of the means below are about 0.02 for the networks and 0.01 for
    the model hedge on the price-basis cost.

    The swap is not requoted. Its mark stays the pricing instrument, so
    the gains and the hedge ratio are unchanged. The cost model takes a
    unit instead. On the price, 50 bp times a mark of about 0.003 barely
    charges a swap. Per contract, 0.002 currency units times a holding of
    a few hundred is a real charge.

    Charged on the mark, the network's mean absolute swap holding is 232
    against 499 for the model hedge. Mean P&L -0.52 against -1.41. CVaR
    1.30 against 1.84. Spot only, with the model hedge's shares and no
    swap, has CVaR 2.84.

    Charged 0.002 per swap on top of 50 bp on the spot, the network's
    mean absolute swap holding is 0.005 (the largest holding on these
    paths is 0.11). Mean P&L -0.31, CVaR 1.43. The model hedge, which
    keeps a large swap and pays the unit charge, has mean -7.34 and CVaR
    10.57. The network's answer on this sample is to drop the swap and
    hold less of the spot (mean absolute spot position 0.15, against 0.33
    when the swap is charged on its mark).

    ## What is differentiable

    The proportional cost and the quadratic cost are the same function in
    training and in evaluation. The absolute value is not differentiable
    at a zero trade. Its subgradient there is 0, which is the convention
    the one-instrument training loss already uses.

    The fixed cost is an indicator. Evaluation uses the indicator. A
    training step uses tanh of the absolute change divided by a width, so
    a dust trade is cheap and a trade of several widths pays the full
    charge, and the step has a gradient. Width 0.01, the library default,
    is already saturated on a daily delta move, so a network that is
    already trading gets no further signal from the fixed cost. The fixed
    cost in this session is trained at width 0.05. The width does not
    enter the plotted P&L.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## Try this\nWrite down a guess before you move the control. Then open the answer."),
        mo.accordion({
            "1. Set the cost to no cost. Where do the black curves sit?": mo.md(
                "On delta. Entropic risk is 0.062 for the network and 0.063 for "
                "delta, on 20 000 paths. The incoming position is irrelevant, so "
                "the three curves lie on top of each other and on the blue line."
            ),
            "2. Compare proportional 50 bp with quadratic impact 0.02. "
            "Which one punishes delta more?": mo.md(
                "The quadratic cost. Delta's entropic risk goes from 1.54 at "
                "50 bp to 2.46 under impact 0.02, because the open and the "
                "close are large trades and the charge squares them. The "
                "network's entropic risk is 1.05 and 0.94. Both still show a "
                "band: the black curves separate."
            ),
            "3. Select the fixed charge of 0.05. Does a band appear?": mo.md(
                "No. At S = 100 the rule returns about 0.55 whether it arrives "
                "holding 0.2, 0.5 or 0.8. It trades on 28 per cent of dates, "
                "delta on 93 per cent, and its entropic risk is 1.79 against "
                "delta's 1.61. Fewer trades, a looser hedge, and no improvement "
                "on the training objective."
            ),
            "4. Switch to Heston. Then switch the swap from the mark to "
            "the per-contract charge. What happens to the red dots?": mo.md(
                "They leave a cloud around a few hundred swaps and sit on zero. "
                "Mean absolute swap holding goes from 232 to 0.005. The model "
                "hedge's mean P&L goes from -1.41 to -7.34, because that book "
                "keeps the swap and now pays 0.002 per contract, including "
                "liquidation."
            ),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
