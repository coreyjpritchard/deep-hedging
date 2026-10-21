import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="03 Whalley-Wilmott band")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from deephedge.band import (
        cloud_half_width,
        describe_rule,
        fit_power_exponent,
        network_positions,
        score_hedgers,
        whalley_wilmott_half_width,
    )
    from deephedge.env import bs_call_delta
    from deephedge.policy import position_grid
    from deephedge.train import (
        DEFAULT,
        WW_COST_GRID,
        WW_FIXED_HALF_WIDTH,
        load_ww_comparison_policy,
    )

    torch.set_num_threads(4)
    return (
        DEFAULT,
        WW_COST_GRID,
        WW_FIXED_HALF_WIDTH,
        bs_call_delta,
        cloud_half_width,
        describe_rule,
        fit_power_exponent,
        go,
        load_ww_comparison_policy,
        mo,
        network_positions,
        np,
        position_grid,
        score_hedgers,
        torch,
        whalley_wilmott_half_width,
    )


@app.cell(hide_code=True)
def _(mo):
    regime = mo.ui.radio(
        options={"30 steps, 25 to 200 bp": 30, "120 steps, 5 to 25 bp": 120},
        value="30 steps, 25 to 200 bp",
        label="Rehedging grid",
    )
    time_left = mo.ui.slider(
        steps=[0.1, 0.25, 0.5, 0.75, 0.9],
        value=0.5,
        show_value=True,
        label="Time to expiry, as a fraction of T",
    )
    feature = mo.ui.radio(
        options={"Session 00 inputs": "mlp", "Session 00 inputs plus gamma": "gamma"},
        value="Session 00 inputs",
        label="Learned rule drawn on the band",
    )
    mo.vstack([
        mo.md(
            "# The learned band against Whalley and Wilmott\n"
            "A short 30-day at-the-money call, Black-Scholes, proportional cost only. "
            "The shade is the asymptotic no-trade band. The dots are positions the "
            "network holds. The black curves are its decision rule."
        ),
        regime,
        time_left,
        feature,
    ])
    return feature, regime, time_left


@app.cell(hide_code=True)
def _(WW_COST_GRID, mo, regime):
    _costs = WW_COST_GRID[regime.value]
    _options = {f"{round(c * 1e4):.0f} bp": c for c in _costs}
    _default = "50 bp" if regime.value == 30 else "10 bp"
    cost = mo.ui.radio(options=_options, value=_default, inline=True, label="Proportional cost")
    cost
    return (cost,)


@app.cell(hide_code=True)
def _(
    DEFAULT,
    WW_COST_GRID,
    WW_FIXED_HALF_WIDTH,
    cloud_half_width,
    describe_rule,
    fit_power_exponent,
    bs_call_delta,
    load_ww_comparison_policy,
    np,
    score_hedgers,
    whalley_wilmott_half_width,
):
    _market = DEFAULT.market()
    _lam = DEFAULT.lam
    _edges = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    rows = {}
    paths = {}
    held_mid = {30: {"mlp": [], "gamma": []}, 120: {"mlp": [], "gamma": []}}
    for _n, _costs in WW_COST_GRID.items():
        _paths = _market.simulate(DEFAULT.m_eval, _n, seed=12345)
        paths[_n] = _paths
        _k = _n // 2
        _tau = float(_market.maturity - _paths.time[_k])
        _spot = _paths.spot[:, _k]
        _delta = bs_call_delta(_spot, _market.strike, _tau, _market.sigma, _market.r)
        _t = float(_paths.time[_k])
        for _cost in _costs:
            _policies = {}
            for _kind in ("mlp", "gamma"):
                _policy, _trained = load_ww_comparison_policy(_n, _cost, _kind)
                if _trained.n_steps != _n or abs(_trained.cost - _cost) > 1e-12:
                    raise RuntimeError(f"weights for {_kind} n={_n} c={_cost} do not match")
                _policies[_kind] = _policy
            _scores, _books = score_hedgers(
                _market,
                _paths,
                _cost,
                _lam,
                _policies,
                WW_FIXED_HALF_WIDTH[(_n, _cost)],
            )
            _snap = {}
            for _kind, _policy in _policies.items():
                _desc = describe_rule(
                    _policy,
                    _t,
                    np.array([100.0]),
                    _market.strike,
                    _market.maturity,
                    _market.sigma,
                    _market.r,
                )
                _column = _books[_kind][:, _k]
                _mids, _emp = [], []
                for _lo, _hi in zip(_edges[:-1], _edges[1:], strict=True):
                    _mask = (_spot >= _lo) & (_spot < _hi)
                    if int(_mask.sum()) < 20:
                        continue
                    _gap = _column[_mask] - _delta[_mask]
                    _q10, _q90 = np.quantile(_gap, [0.1, 0.9])
                    _mids.append(0.5 * (_lo + _hi))
                    _emp.append(0.5 * (_q90 - _q10))
                _mids_a = np.asarray(_mids)
                _emp_a = np.asarray(_emp)
                _ww_bins = whalley_wilmott_half_width(
                    _mids_a, _tau, _cost, _lam, _market.strike, _market.sigma, _market.r
                )
                _snap[_kind] = {
                    "centre_offset": float(_desc["centre_offset"][0]),
                    "slope": float(_desc["slope"][0]),
                    "no_trade": float(_desc["no_trade_half_width"][0]),
                    "cloud": cloud_half_width(_column, _spot, _delta, 99.0, 101.0),
                    "spot_corr": float(np.corrcoef(_emp_a, _ww_bins)[0, 1]),
                    "spot_ratio": float(np.median(_emp_a / _ww_bins)),
                }
                held_mid[_n][_kind].append(_column.copy())
            _ww_atm = float(
                whalley_wilmott_half_width(
                    100.0, _tau, _cost, _lam, _market.strike, _market.sigma, _market.r
                )
            )
            rows[(_n, _cost)] = {
                "scores": _scores,
                "snap": _snap,
                "policies": _policies,
                "fixed_w": WW_FIXED_HALF_WIDTH[(_n, _cost)],
                "ww_atm": _ww_atm,
                "spot_mid": _spot,
                "delta_mid": _delta,
            }
    fits = {}
    _rng = np.random.default_rng(0)
    for _n, _costs in WW_COST_GRID.items():
        _k = _n // 2
        _spot = rows[(_n, _costs[0])]["spot_mid"]
        _delta = rows[(_n, _costs[0])]["delta_mid"]
        # The analytic half-width at the money, not a cloud of WW holdings.
        _ww = [rows[(_n, _c)]["ww_atm"] for _c in _costs]
        _ww_fit = fit_power_exponent(_costs, _ww)
        fits[(_n, "ww", "analytic")] = {
            "exponent": _ww_fit["exponent"],
            "lack_of_fit_se": _ww_fit["se"],
            "bootstrap_se": 0.0,
            "widths": _ww,
        }
        for _kind in ("mlp", "gamma"):
            for _measure in ("cloud", "no_trade"):
                _widths = [rows[(_n, _c)]["snap"][_kind][_measure] for _c in _costs]
                _point = fit_power_exponent(_costs, _widths)
                _boot = []
                if _measure == "cloud":
                    _columns = held_mid[_n][_kind]
                    for _ in range(400):
                        _idx = _rng.integers(0, _spot.size, size=_spot.size)
                        _draw = [
                            cloud_half_width(
                                _col[_idx], _spot[_idx], _delta[_idx], 99.0, 101.0
                            )
                            for _col in _columns
                        ]
                        _boot.append(fit_power_exponent(_costs, _draw)["exponent"])
                    _boot_se = float(np.std(_boot, ddof=1))
                else:
                    _boot_se = float("nan")
                fits[(_n, _kind, _measure)] = {
                    "exponent": _point["exponent"],
                    "lack_of_fit_se": _point["se"],
                    "bootstrap_se": _boot_se,
                    "widths": _widths,
                }
    book = {"rows": rows, "paths": paths, "fits": fits, "market": _market}
    return (book,)


@app.cell(hide_code=True)
def _(
    DEFAULT,
    book,
    bs_call_delta,
    cost,
    describe_rule,
    feature,
    go,
    mo,
    network_positions,
    np,
    position_grid,
    regime,
    time_left,
    whalley_wilmott_half_width,
):
    _market = book["market"]
    _n = regime.value
    _cost = cost.value
    _kind = feature.value
    _row = book["rows"][(_n, _cost)]
    _paths = book["paths"][_n]
    _frac = time_left.value
    _t = _market.maturity * (1.0 - _frac)
    _tau = _market.maturity - _t
    _k = int(np.argmin(np.abs(_paths.time - _t)))
    _k = min(_k, _n - 1)
    _policy = _row["policies"][_kind]
    _positions = network_positions(_policy, _paths)
    _spot_k = _paths.spot[:, _k]
    _held = _positions[:, _k]
    _show = np.random.default_rng(0).choice(_spot_k.size, 3_000, replace=False)
    _spots = np.linspace(85.0, 115.0, 121)
    _delta = bs_call_delta(_spots, _market.strike, _tau, _market.sigma, _market.r)
    _width = whalley_wilmott_half_width(
        _spots, _tau, _cost, DEFAULT.lam, _market.strike, _market.sigma, _market.r
    )
    _rule = describe_rule(
        _policy,
        _t,
        _spots,
        _market.strike,
        _market.maturity,
        _market.sigma,
        _market.r,
    )
    _prevs = np.array([0.2, 0.5, 0.8])
    _grid = position_grid(_policy, _t, _spots, _prevs)
    _label = "session 00 inputs" if _kind == "mlp" else "session 00 inputs plus gamma"
    _colours = {"band": "#1f77b4", "net": "#d62728", "rule": "#2b2b2b"}

    fig_band = go.Figure()
    fig_band.add_scatter(
        x=_spots, y=_delta + _width, mode="lines", line={"width": 0},
        showlegend=False, hoverinfo="skip",
    )
    fig_band.add_scatter(
        x=_spots, y=_delta - _width, mode="lines", line={"width": 0},
        fill="tonexty", name="Whalley-Wilmott band",
        fillcolor="rgba(31, 119, 180, 0.18)", hoverinfo="skip",
    )
    fig_band.add_scatter(
        x=_spots, y=_delta, name="Black-Scholes delta",
        line={"color": _colours["band"], "width": 2.5},
    )
    fig_band.add_scatter(
        x=_spots, y=_rule["centre"], name="learned centre",
        line={"color": _colours["net"], "width": 2.5},
    )
    fig_band.add_scattergl(
        x=_spot_k[_show], y=_held[_show], mode="markers",
        marker={"color": _colours["net"], "size": 3, "opacity": 0.25},
        name="position held on each path", hoverinfo="skip",
    )
    for _p, _y, _dash in zip(_prevs, _grid, ["dot", "dash", "dashdot"], strict=True):
        fig_band.add_scatter(
            x=_spots, y=_y, name=f"new position if it holds {_p:.1f}",
            line={"color": _colours["rule"], "dash": _dash, "width": 1.6},
        )
    _days = _frac * _market.maturity * 365
    fig_band.update_layout(
        title=(
            f"(a) Band with {_days:.1f} days left, cost {_cost * 1e4:.0f} bp, "
            f"{_n} steps, {_label}"
        ),
        xaxis_title="spot S",
        xaxis_range=[85, 115],
        yaxis_title="position in the underlying",
        yaxis_range=[-0.2, 1.2],
        legend={"orientation": "h", "y": -0.25},
        height=520,
        margin={"t": 60, "b": 10},
    )

    def _money(score, key):
        body = f"{score[key]:+.3f}" if key == "mean" else f"{score[key]:.3f}"
        return f"{body} ({score[key + '_se']:.3f})"

    _names = [
        ("delta", "Delta"),
        ("ww", "Whalley-Wilmott"),
        ("fixed", f"Fixed width {_row['fixed_w']:.2f}"),
        ("mlp", "Network"),
        ("gamma", "Network with gamma"),
    ]
    _lines = [
        "| hedger | mean P&L (SE) | entropic risk (SE) |",
        "|---|---:|---:|",
    ]
    for _key, _name in _names:
        _sc = _row["scores"][_key]
        _lines.append(f"| {_name} | {_money(_sc, 'mean')} | {_money(_sc, 'entropic')} |")
    _snap = _row["snap"][_kind]
    table = mo.md(
        "\n".join(_lines)
        + "\n\n"
        + (
            f"Same {DEFAULT.m_eval:,} paths, seed 12345, for every row. "
            f"At S = 100 and t = T/2 the Whalley-Wilmott half-width is "
            f"{_row['ww_atm']:.3f} shares. For the rule drawn above, the cloud is "
            f"{_snap['cloud']:.3f}, the centre sits {_snap['centre_offset']:+.3f} from "
            f"delta, and the slope there is {_snap['slope']:.2f}. "
            f"The fixed width was chosen on seed 0."
        )
    )

    fig_scale = go.Figure()
    _grid_costs = np.geomspace(0.0005, 0.02, 40)
    _tau_mid = _market.maturity / 2
    _ww_curve = np.array([
        float(whalley_wilmott_half_width(
            100.0, _tau_mid, float(_c), DEFAULT.lam, _market.strike, _market.sigma, _market.r
        ))
        for _c in _grid_costs
    ])
    fig_scale.add_scatter(
        x=_grid_costs * 1e4, y=_ww_curve, name="Whalley-Wilmott at S = 100, t = T/2",
        line={"color": "#1f77b4", "width": 2.5},
    )
    _anchor = book["rows"][(120, 0.0005)]["snap"]["mlp"]["cloud"]
    fig_scale.add_scatter(
        x=_grid_costs * 1e4,
        y=_anchor * (_grid_costs / 0.0005) ** (1.0 / 3.0),
        name="cube root through the 120-step cloud at 5 bp",
        line={"color": "#1f77b4", "width": 1.5, "dash": "dot"},
    )
    _style = {
        (30, "mlp"): {
            "name": "30 steps, session 00 inputs",
            "color": "#d62728",
            "symbol": "circle",
        },
        (120, "mlp"): {
            "name": "120 steps, session 00 inputs",
            "color": "#d62728",
            "symbol": "diamond",
        },
        (30, "gamma"): {
            "name": "30 steps, plus gamma",
            "color": "#2ca02c",
            "symbol": "circle-open",
        },
        (120, "gamma"): {
            "name": "120 steps, plus gamma",
            "color": "#2ca02c",
            "symbol": "diamond-open",
        },
    }
    for (_n_i, _kind_i), _st in _style.items():
        _cs = [c for (_nn, c) in book["rows"] if _nn == _n_i]
        _ws = [book["rows"][(_n_i, c)]["snap"][_kind_i]["cloud"] for c in _cs]
        fig_scale.add_scatter(
            x=np.asarray(_cs) * 1e4, y=_ws, mode="lines+markers", name=_st["name"],
            line={"color": _st["color"], "width": 1.5},
            marker={"symbol": _st["symbol"], "size": 9},
        )
    fig_scale.update_layout(
        title="(b) Half-width against cost, at the money, halfway to expiry",
        xaxis_title="proportional cost (basis points)",
        yaxis_title="half-width (shares)",
        xaxis_type="log",
        yaxis_type="log",
        legend={"orientation": "h", "y": -0.25},
        height=460,
        margin={"t": 60, "b": 10},
    )

    fig_centre = go.Figure()
    fig_centre.add_hline(y=0.0, line={"color": "#1f77b4", "width": 2}, annotation_text="delta")
    for (_n_i, _kind_i), _st in _style.items():
        _cs = [c for (_nn, c) in book["rows"] if _nn == _n_i]
        _off = [book["rows"][(_n_i, c)]["snap"][_kind_i]["centre_offset"] for c in _cs]
        fig_centre.add_scatter(
            x=np.asarray(_cs) * 1e4, y=_off, mode="lines+markers", name=_st["name"],
            line={"color": _st["color"], "width": 1.5},
            marker={"symbol": _st["symbol"], "size": 9},
        )
    fig_centre.update_layout(
        title="(c) Learned centre minus delta, at S = 100 and t = T/2",
        xaxis_title="proportional cost (basis points)",
        yaxis_title="centre offset (shares)",
        xaxis_type="log",
        legend={"orientation": "h", "y": -0.25},
        height=420,
        margin={"t": 60, "b": 10},
    )
    return fig_band, fig_centre, fig_scale, table


@app.cell(hide_code=True)
def _(fig_band, fig_centre, fig_scale, mo, table):
    mo.vstack([fig_band, table, fig_scale, fig_centre])
    return


@app.cell(hide_code=True)
def _(book, mo):
    def _fit(n_steps, kind, measure):
        return book["fits"][(n_steps, kind, measure)]

    def _exp(n_steps, kind):
        fit = _fit(n_steps, kind, "cloud")
        return (
            f"{fit['exponent']:.3f} "
            f"(lack-of-fit SE {fit['lack_of_fit_se']:.3f}, "
            f"bootstrap SE {fit['bootstrap_se']:.3f})"
        )

    _slopes_30 = [
        book["rows"][(30, c)]["snap"]["mlp"]["slope"] for c in (0.0025, 0.005, 0.01, 0.02)
    ]
    _slopes_120 = [
        book["rows"][(120, c)]["snap"]["mlp"]["slope"] for c in (0.0005, 0.001, 0.0015, 0.0025)
    ]
    _g_slopes = [
        book["rows"][(n, c)]["snap"]["gamma"]["slope"]
        for n, costs in ((30, (0.005,)), (120, (0.001,)))
        for c in costs
    ]
    _off_200 = book["rows"][(30, 0.02)]["snap"]["mlp"]["centre_offset"]
    _off_5 = book["rows"][(120, 0.0005)]["snap"]["mlp"]["centre_offset"]
    _corr_50 = book["rows"][(30, 0.005)]["snap"]["mlp"]["spot_corr"]
    _ratio_50 = book["rows"][(30, 0.005)]["snap"]["mlp"]["spot_ratio"]
    _corr_g = book["rows"][(30, 0.005)]["snap"]["gamma"]["spot_corr"]
    _ratio_g = book["rows"][(30, 0.005)]["snap"]["gamma"]["spot_ratio"]
    _nt = _fit(30, "mlp", "no_trade")
    _nt120 = _fit(120, "mlp", "no_trade")
    _ww_fit = _fit(120, "ww", "analytic")
    _mean_costs = ((120, 0.0005), (120, 0.0025), (30, 0.005), (30, 0.02))
    _mean_ok = all(
        book["rows"][key]["scores"][name]["mean"] > book["rows"][key]["scores"]["delta"]["mean"]
        for key in _mean_costs
        for name in ("ww", "fixed", "mlp", "gamma")
    )
    _mean_line = (
        "Mean P&L is less negative for every band than for delta, on each of these "
        "four costs. Hypothesis 5 holds on the rows above."
        if _mean_ok
        else "Mean P&L is not less negative for every band than for delta. "
        "Hypothesis 5 does not hold on every row above."
    )

    def _row_sentence(n_steps, cost, label):
        scores = book["rows"][(n_steps, cost)]["scores"]
        order = sorted(
            ("ww", "fixed", "mlp", "gamma"),
            key=lambda name: scores[name]["entropic"],
        )
        names = {
            "ww": "Whalley-Wilmott",
            "fixed": "fixed width",
            "mlp": "network",
            "gamma": "gamma network",
        }
        ranking = ", then ".join(names[name] for name in order)
        delta = scores["delta"]["entropic"]
        gap = scores[order[1]]["entropic"] - scores[order[0]]["entropic"]
        se = (
            scores[order[0]]["entropic_se"] ** 2 + scores[order[1]]["entropic_se"] ** 2
        ) ** 0.5
        tie = (
            f" The first two differ by {gap:.3f}, against a combined standard error of {se:.3f}."
        )
        return (
            f"At {label}, ordered by entropic risk, the bands are {ranking}. "
            f"Delta is {delta:.3f}.{tie}"
        )

    _f120 = _fit(120, "mlp", "cloud")
    _f30 = _fit(30, "mlp", "cloud")
    _gap120 = _f120["exponent"] - (1.0 / 3.0)
    _gap30 = _f30["exponent"] - (1.0 / 3.0)
    _boot_units = abs(_gap120) / _f120["bootstrap_se"]
    _lof_units = abs(_gap120) / _f120["lack_of_fit_se"]
    _boot30 = abs(_gap30) / _f30["bootstrap_se"]
    _lof30 = abs(_gap30) / _f30["lack_of_fit_se"]

    mo.md(
        f"""
## What the three panels show

Panel (a) is one date. The blue shade is the set of holdings Whalley and Wilmott
tell you not to trade. Its centre is the Black-Scholes delta and its half-width
depends on the spot and on the time left. The red dots are where the network's
holding actually sits on each path. The red curve is the holding it leaves
unchanged. The three black curves are the decision rule: fix the incoming
position at 0.2, 0.5 or 0.8 and read off the position it chooses. A sharp band
would leave those curves on top of each other inside the shade and on the edge
outside it. A soft rule keeps them apart.

Panel (b) is the half-width at S = 100 and t = T/2, against the cost, on a log
scale. The Whalley-Wilmott curve is exactly a cube root. The dotted line is the
same cube root, scaled to pass through the 120-step cloud at 5 bp, so a parallel
cloud is a cube root of a different level. The cloud is half the distance from
the 10th to the 90th percentile of holding minus delta, for paths with spot
between 99 and 101. That is the width session 00 was looking at.

Panel (c) is the learned centre minus delta at the same point. The asymptotic
centre is zero on this plot, because the drift equals the interest rate.

The table uses one batch of {20000:,} paths, seed 12345, sold at the Black-Scholes
premium. Every hedger pays the proportional cost on the open and on the
liquidation. The fixed width is the constant, on a grid of 0.02 shares, that
minimised entropic risk on a different batch (seed 0). It has one tuned
parameter. The asymptotic band has none.

## The formula

Whalley and Wilmott (1997, Mathematical Finance 7(3), equation (17) and section
4) study a small proportional cost. A trade of N shares loses epsilon times N
times S. That is `ProportionalCost`. Their utility is 1 minus exp of minus gamma
times terminal wealth, so gamma is absolute risk aversion in units of one over
a unit of wealth. Maximising that utility is the same problem as minimising the
entropic risk of terminal wealth at aversion gamma. This package's entropic risk
is applied to the time-0 P&L. At r = 0 the time-0 P&L is the change in wealth,
so their gamma equals our lambda. At r > 0 the two match when lambda equals
gamma times exp(r T). This session has r = 0 and lambda = 1, so the risk
aversion in the formula is 1.

When the real-world drift equals r, the centre is the Black-Scholes delta and
the half-width, in shares, is

$$
\\left( \\frac{{3}}{{2}}\\,
\\frac{{\\varepsilon\\, S\\, e^{{-r(T-t)}}\\, \\Gamma^2}}{{\\gamma}}
\\right)^{{1/3}}.
$$

Gamma here is the option's gamma. The combination epsilon S Gamma squared over
gamma is dimensionless, and so is the cube root. A drift different from r would
add a Merton term to the centre. This market has mu = r = 0, so that term is
zero and is not applied. Inside the band the hedger does not trade. Outside it,
the hedger trades to the nearer edge, not to delta.

## Hypotheses, stated before the runs

1. The analytic half-width scales as the cube root of the cost.
2. The learned cloud at the money, at t = T/2, scales close to that cube root
   at 120 steps and 5 to 25 bp, and closer to flat than to a cube root at 30
   steps and 25 to 200 bp.
3. The learned centre sits at or above delta, and the offset grows with the
   cost. The asymptotic centre is delta.
4. Every band beats delta on entropic risk when the cost is positive. At small
   cost the asymptotic band should match or beat the tuned constant width, and
   the network should be close to both. At large cost the network may beat the
   asymptotic band, because a centre shift is outside the approximation.
5. Mean P&L of a band is less negative than delta's, because delta trades more.
6. The previous position is already an input, and time and log moneyness already
   determine gamma. Adding gamma as a fourth input is a coordinate change. If
   the network has missed the shape because it lacked that coordinate, the
   correlation of width with gamma across the spot should rise and the cost
   exponent should move toward 1/3. If the miss is the soft optimum, the two
   networks should agree.

## What this sample says

The analytic exponent is {_ww_fit['exponent']:.2f}, with lack-of-fit standard
error {_ww_fit['lack_of_fit_se']:.2f}. That is the identity in hypothesis 1, on
both grids. It is the exponent of the formula at the money, not of a cloud.

The 120-step cloud, session 00 inputs, has exponent {_exp(120, "mlp")}. The
gap from 1/3 is {_gap120:+.3f}, which is {_boot_units:.1f} bootstrap standard
errors and {_lof_units:.1f} lack-of-fit standard errors. The bootstrap standard
error is the sampling error of that slope on these 20 000 paths. The lack-of-fit
standard error is the ordinary regression formula, which treats the four
residuals as noise. The widths stay close to the dotted cube root in panel (b),
and the slope is still distinguishable from 1/3 because the cloud is measured
precisely. The 30-step cloud has exponent {_exp(30, "mlp")}. The gap from 1/3
is {_gap30:+.3f}, or {_boot30:.0f} bootstrap standard errors and {_lof30:.1f}
lack-of-fit standard errors. Those widths are flat. With gamma as an input the
exponents are {_exp(120, "gamma")} at 120 steps and {_exp(30, "gamma")} at 30
steps. That input does not move the cloud onto a cleaner cube root.

The analytic 1/3 is the wrong yardstick for a cloud on a discrete grid. The
same cloud measured on the Whalley-Wilmott hedger, which trades to the edge of
the analytic band on these paths, has exponent 0.227 at 120 steps
(lack-of-fit SE 0.033) and 0.041 at 30 steps (lack-of-fit SE 0.007). So the
network's 120-step cloud scales more like a cube root than the asymptotic
rule's own cloud does, and the flat 30-step cloud is what the asymptotic rule
shows on that grid too. The gaps from 1/3 above measure the grid as well as
the network. These two reference values come from a one-off check outside
this session; the cells above do not compute them.

The no-trade half-width, defined as the largest deviation from the centre that
still trades less than 0.005 shares, is a different object. Its exponent is
{_nt['exponent']:.2f} (lack-of-fit SE {_nt['lack_of_fit_se']:.2f}) at 30 steps
and {_nt120['exponent']:.2f} (lack-of-fit SE {_nt120['lack_of_fit_se']:.2f}) at
120 steps. A steeper soft slope stays inside that absolute tolerance farther
from the centre, so this exponent can be large while the cloud is flat. It is
not the cube-root test.

Across spot, at 30 steps and 50 bp, the cloud in bins from 85 to 115 correlates
{_corr_50:.3f} with the asymptotic width, at {_ratio_50:.2f} of its level. With
the gamma input the correlation is {_corr_g:.3f} and the ratio is
{_ratio_g:.2f}. That correlation does not show that the learned band has the
gamma shape. On the same paths the constant-width band, whose half-width does
not depend on the spot, has a cloud that correlates 0.974 to 0.998 with the
asymptotic width across the eight rows, because delta moves by about gamma
times the spot move between dates and every band's cloud follows that. The
level is not a band's level either: the Whalley-Wilmott hedger's own cloud is
0.59 to 0.92 of its analytic half-width, and the network's cloud is 0.50 to
0.64 of the Whalley-Wilmott hedger's. These comparisons come from the same
one-off check. At 30 steps the network's cloud does not grow like a cube root,
and neither does the Whalley-Wilmott hedger's.

The slope of the decision rule at the centre runs from {_slopes_30[0]:.2f} to
{_slopes_30[-1]:.2f} on the 30-step grid and from {_slopes_120[0]:.2f} to
{_slopes_120[-1]:.2f} on the 120-step grid. A sharp band would have slope 1
inside and 0 outside. These slopes are the soft shrinkage session 00 already
found. The gamma-input slopes at 50 bp (30 steps) and 10 bp (120 steps) are
{_g_slopes[0]:.2f} and {_g_slopes[1]:.2f}.

The centre offset at 5 bp and 120 steps is {_off_5:+.3f} shares. At 200 bp and
30 steps it is {_off_200:+.3f}. Hypothesis 3 holds on the large-cost grid and
is small on the small-cost grid, which is where the asymptotic centre is doing
its job.

{_row_sentence(120, 0.0005, "120 steps and 5 bp")}
{_row_sentence(120, 0.0025, "120 steps and 25 bp")}
{_row_sentence(30, 0.005, "30 steps and 50 bp")}
{_row_sentence(30, 0.02, "30 steps and 200 bp")}

{_mean_line}

The gamma input does not produce a sharp band and does not move the cloud onto
the asymptotic level. The previous position was already a feature. Gamma was
already a function of the other two. The remaining gap is the soft rule and,
at large cost, a centre the asymptotic formula does not have.

The next comparison is the same four hedgers under Heston, and then under rough
Bergomi using the `roughvol` package from the rough-volatility repository.
"""
    )
    return


if __name__ == "__main__":
    app.run()
