import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Playground")


@app.cell(hide_code=True)
def _():
    import html
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch

    from deephedge.bootstrap import parse_closes
    from deephedge.playground import (
        CARD_NAMES,
        HEDGER_NAMES,
        PlaygroundError,
        PlaygroundSpec,
        run_experiment,
        spec_id,
    )
    from deephedge.upload import (
        MarketUploadError,
        decode_text,
        load_uploaded_market,
        validate_market,
    )

    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    example_text = (root / "examples" / "daily_closes.csv").read_text()
    example_closes = parse_closes(example_text)
    template_text = (root / "templates" / "merton_jump.py").read_text()
    return (
        CARD_NAMES,
        HEDGER_NAMES,
        MarketUploadError,
        PlaygroundError,
        PlaygroundSpec,
        decode_text,
        example_closes,
        example_text,
        go,
        html,
        load_uploaded_market,
        mo,
        np,
        parse_closes,
        run_experiment,
        spec_id,
        template_text,
        validate_market,
    )


@app.cell(hide_code=True)
def _(go, np):
    def loss_figure(history, title):
        fig = go.Figure()
        fig.add_scatter(
            x=list(range(1, len(history) + 1)),
            y=list(history),
            mode="lines+markers",
            name="entropic risk",
            line={"color": "#d62728"},
        )
        fig.update_layout(
            title=title,
            xaxis_title="epoch",
            yaxis_title="entropic risk of training P&L",
            height=360,
            margin={"t": 60, "b": 40},
        )
        return fig

    def histogram_figure(result):
        series = [("Train, this hedge", result.pnl_train_chosen, "#ff7f0e")]
        if np.allclose(result.pnl_test_chosen, result.pnl_test_delta):
            series.append(("Test, delta (this hedge is delta)", result.pnl_test_delta, "#1f77b4"))
        else:
            series.append(("Test, this hedge", result.pnl_test_chosen, "#d62728"))
            series.append(("Test, delta", result.pnl_test_delta, "#1f77b4"))
        pooled = np.concatenate([values for _name, values, _colour in series])
        edges = np.linspace(np.quantile(pooled, 0.005), np.quantile(pooled, 0.995), 61)
        centres = 0.5 * (edges[1:] + edges[:-1])
        width = float(edges[1] - edges[0])
        fig = go.Figure()
        for name, values, colour in series:
            density, _ = np.histogram(values, bins=edges, density=True)
            fig.add_bar(
                x=centres, y=density, width=width, name=name,
                marker_color=colour, opacity=0.55,
            )
        fig.update_layout(
            barmode="overlay",
            title="Terminal P&L. The premium is the training price on every sample.",
            xaxis_title="P&L (premium - discounted payoff + gains - costs)",
            yaxis_title="density",
            legend={"orientation": "h", "y": -0.25},
            height=420,
            margin={"t": 60, "b": 10},
        )
        return fig

    def path_figure(result):
        time = result.sample_time
        fig = go.Figure()
        fig.add_scatter(x=time, y=result.sample_spot, name="spot", line={"color": "#333333"})
        fig.add_scatter(
            x=time[:-1], y=result.sample_chosen, name=result.hedger_name,
            yaxis="y2", line={"color": "#d62728"},
        )
        fig.add_scatter(
            x=time[:-1], y=result.sample_delta, name="delta",
            yaxis="y2", line={"color": "#1f77b4", "dash": "dash"},
        )
        fig.update_layout(
            title=f"One test path, seed 12345, first draw. Hedge: {result.hedger_name}.",
            xaxis_title="time in years",
            yaxis={"title": "spot"},
            yaxis2={"title": "shares", "overlaying": "y", "side": "right"},
            legend={"orientation": "h", "y": -0.25},
            height=400,
            margin={"t": 60, "b": 10},
        )
        return fig

    def band_figure(result):
        spots = result.band_spots
        delta = result.band_delta
        half = result.band_half_width
        fig = go.Figure()
        if result.ww_applies:
            fig.add_scatter(
                x=np.concatenate([spots, spots[::-1]]),
                y=np.concatenate([delta - half, (delta + half)[::-1]]),
                fill="toself",
                name="Whalley-Wilmott band",
                mode="lines",
                line={"width": 0},
                fillcolor="rgba(31, 119, 180, 0.18)",
            )
        fig.add_scatter(
            x=spots, y=delta, name="delta",
            line={"color": "#1f77b4", "dash": "dash"},
        )
        colours = ("#d62728", "#2ca02c")
        for i, prev in enumerate(result.band_prevs):
            fig.add_scatter(
                x=spots, y=result.band_response[i],
                name=f"hedge, already holding {prev:.1f}",
                line={"color": colours[i]},
            )
        fig.update_layout(
            title="Halfway to expiry on the test market. Shares against the spot.",
            xaxis_title="spot",
            yaxis_title="shares",
            legend={"orientation": "h", "y": -0.25},
            height=440,
            margin={"t": 60, "b": 10},
        )
        return fig

    def stats_table(result):
        def row(name, stats):
            return (
                f"| {name} | {stats['mean']:+.3f} ({stats['mean_se']:.3f}) "
                f"| {stats['entropic']:.3f} ({stats['entropic_se']:.3f}) "
                f"| {stats['cvar']:.3f} |"
            )
        lines = [
            "| Sample | mean (SE) | entropic (SE) | CVaR |",
            "|---|---:|---:|---:|",
            row(f"Train, {result.hedger_name}", result.train_chosen),
            row("Train, delta", result.train_delta),
            row(f"Test, {result.hedger_name}", result.test_chosen),
            row("Test, delta", result.test_delta),
        ]
        m = result.test_chosen["m"]
        note = (
            f"\n\nSample size {m}. Standard errors are for the mean and for the "
            "entropic risk on this sample. CVaR has no standard error in this table. "
            f"Training premium {result.premium_train:.3f}. "
            f"The test market's own premium is {result.premium_test:.3f}."
        )
        return "\n".join(lines) + note

    return band_figure, histogram_figure, loss_figure, path_figure, stats_table


@app.cell(hide_code=True)
def _(mo):
    cards = {
        "Black-Scholes": "bs",
        "Heston": "heston",
        "Rough Bergomi": "rough",
        "CSV bootstrap": "csv",
        "Python market": "py",
    }
    train_card = mo.ui.radio(options=cards, value="Black-Scholes", label="Train in")
    test_card = mo.ui.radio(options=cards, value="Heston", label="Test in")
    s0 = mo.ui.number(1, 1000, step=1, value=100, label="Spot s0")
    strike = mo.ui.number(1, 1000, step=1, value=100, label="Strike K")
    maturity_days = mo.ui.number(1, 365, step=1, value=30, label="Maturity in days")
    rate = mo.ui.number(-0.2, 0.2, step=0.01, value=0.0, label="Interest rate r")
    sigma = mo.ui.number(0.01, 1.0, step=0.01, value=0.2, label="Black-Scholes volatility")
    mu = mo.ui.number(-0.5, 0.5, step=0.01, value=0.0, label="Simulation drift mu")
    kappa = mo.ui.number(0.1, 5.0, step=0.1, value=1.0, label="kappa, variance speed")
    theta = mo.ui.number(0.01, 0.25, step=0.01, value=0.04, label="theta, long-run variance")
    xi = mo.ui.number(0.0, 4.0, step=0.1, value=2.0, label="xi, volatility of variance")
    rho = mo.ui.number(-1.0, 1.0, step=0.05, value=-0.7, label="rho, spot and variance correlation")
    v0 = mo.ui.number(0.01, 0.25, step=0.01, value=0.04, label="v0, variance at inception")
    hurst = mo.ui.number(0.01, 0.49, step=0.01, value=0.1, label="Hurst index H")
    eta = mo.ui.number(0.1, 3.0, step=0.1, value=1.5, label="eta, variance volatility")
    bergomi_rho = mo.ui.number(-1.0, 1.0, step=0.05, value=-0.7, label="rho")
    bergomi_xi = mo.ui.number(0.01, 2.0, step=0.05, value=0.3, label="xi")
    block = mo.ui.number(1, 40, step=1, value=5, label="Block length in returns")
    demean = mo.ui.checkbox(value=False, label="Subtract the mean log return before stitching")
    hedger = mo.ui.radio(
        options={
            "Delta": "delta",
            "Whalley-Wilmott band": "ww",
            "Fixed band": "fixed",
            "Feed-forward network": "mlp",
            "Recurrent network": "gru",
        },
        value="Feed-forward network",
        label="Hedge",
    )
    cost_family = mo.ui.radio(
        options={
            "Proportional": "proportional",
            "Fixed plus proportional": "fixed",
            "Quadratic": "quadratic",
        },
        value="Proportional",
        label="Cost",
    )
    cost_bp = mo.ui.slider(
        0, 200, step=5, value=50, show_value=True,
        label="Proportional cost, basis points of the spot",
    )
    fixed_cash = mo.ui.number(
        0.0, 1.0, step=0.01, value=0.01, label="Fixed cash per instrument per trade date",
    )
    impact = mo.ui.number(0.0, 1.0, step=0.01, value=0.01, label="Quadratic impact")
    half_width = mo.ui.number(
        0.0, 1.0, step=0.01, value=0.10, label="Fixed band half-width, in shares",
    )
    n_paths = mo.ui.dropdown(
        options={"500": 500, "1 000": 1000, "2 000": 2000, "4 000": 4000},
        value="2 000",
        label="Paths in the training sample and in the test sample",
    )
    n_steps = mo.ui.dropdown(
        options={"10": 10, "20": 20, "30": 30, "60": 60},
        value="30",
        label="Rehedging steps",
    )
    epochs = mo.ui.dropdown(
        options={"10": 10, "20": 20, "40": 40},
        value="20",
        label="Epochs, for a network",
    )
    csv_train = mo.ui.file(
        filetypes=[".csv"], kind="area", label="CSV for the training slot",
    )
    csv_test = mo.ui.file(
        filetypes=[".csv"], kind="area", label="CSV for the test slot",
    )
    py_train = mo.ui.file(
        filetypes=[".py"], kind="area", label="Python market for the training slot",
    )
    py_test = mo.ui.file(
        filetypes=[".py"], kind="area", label="Python market for the test slot",
    )
    run = mo.ui.run_button(label="Train and score")
    return (
        bergomi_rho, bergomi_xi, block, cost_bp, cost_family, csv_test, csv_train,
        demean, epochs, eta, fixed_cash, half_width, hedger, hurst, impact, kappa,
        maturity_days, mu, n_paths, n_steps, py_test, py_train, rate, rho, run, s0,
        sigma, strike, test_card, theta, train_card, v0, xi,
    )


@app.cell(hide_code=True)
def _(
    bergomi_rho, bergomi_xi, block, cost_bp, cost_family, csv_test, csv_train,
    demean, epochs, eta, example_text, fixed_cash, half_width, hedger, html,
    hurst, impact, kappa, maturity_days, mo, mu, n_paths, n_steps, py_test,
    py_train, rate, rho, run, s0, sigma, strike, template_text, test_card, theta,
    train_card, v0, xi,
):
    mo.vstack([
        mo.md(
            "# Train in one market, test in another\n"
            "A short call is sold and hedged by trading the spot. "
            "**Train in** draws the paths the hedge is fit on. "
            "**Test in** draws the paths it is scored on."
        ),
        mo.hstack([
            mo.vstack([train_card, mo.md("Paths the hedge is fit on.")]),
            mo.vstack([test_card, mo.md("Paths the hedge is scored on.")]),
        ], widths="equal"),
        mo.md(
            "The slots are radio cards. A drag-and-drop board would be a JavaScript "
            "widget, and this playground is Python so every market, cost and network "
            "is the `deephedge` code the sessions use. "
            "A file can still be dropped on the CSV and Python boxes below. "
            "Reading that file is Python."
        ),
        mo.md(
            "**Black-Scholes** is a constant volatility. "
            "**Heston** is a stochastic variance. The hedge still trades only the spot. "
            "Delta there is the Black-Scholes delta with volatility equal to the square "
            "root of v0, the variance at inception. "
            "**Rough Bergomi** is a card for the roughvol package. It is not simulated "
            "yet, and choosing it shows that message. "
            "**CSV bootstrap** stitches blocks of historical log returns. It is a "
            "data-driven generator, the simplest stand-in for the market generators "
            "of Buehler, Horvath, Lyons, Perez Arribas and Wood (2020). "
            "**Python market** is a file that implements `Market`."
        ),
        mo.callout(
            mo.md(
                "A Python upload is executed on this computer. There is no sandbox. "
                "Use this app locally, and only with a file you trust. "
                "The static page does not run uploads."
            ),
            kind="warn",
        ),
        mo.md("## Contract"),
        mo.md(
            "These four numbers are the call and the discount on Black-Scholes, "
            "Heston, rough Bergomi and the CSV. A Python market uses the spot, "
            "strike, maturity and rate written in the file."
        ),
        s0, strike, maturity_days, rate,
        mo.accordion({
            "Black-Scholes": mo.vstack([
                sigma, mu,
                mo.md(
                    "Sigma is the volatility in the price, the delta and the paths. "
                    "Mu is the drift of the paths. The price uses r, not mu. "
                    "The preset is sigma 20 percent and mu equal to r, which is 0."
                ),
            ]),
            "Heston": mo.vstack([
                kappa, theta, xi, rho, v0,
                mo.md(
                    "The preset is the 2019 experiment: kappa 1, theta 0.04, xi 2, "
                    "rho -0.7, v0 0.04. Mu above is the spot drift. "
                    "The variance swap is priced by the market and is not traded here."
                ),
            ]),
            "Rough Bergomi": mo.vstack([
                hurst, eta, bergomi_rho, bergomi_xi,
                mo.md(
                    "H near 0.1 is a rough path. Eta, rho, xi and v0 are the usual "
                    "Bergomi parameters. v0 is the Heston field above. "
                    "The simulator is the roughvol package, which this card does not call yet."
                ),
            ]),
            "CSV bootstrap": mo.vstack([
                block, demean, csv_train, csv_test,
                mo.md(
                    "Each step of a path is one log return from the file. "
                    "With 30 steps and a 30-day maturity, thirty daily returns sit on "
                    "a 30/365-year grid. The price and the delta use a volatility of "
                    "the return standard deviation times the square root of 252, so "
                    "the path variance and the Black-Scholes variance differ by the "
                    "365-versus-252 day count. "
                    "Leave the mean in the returns to keep the historical drift. "
                    "The discounted spot is then not a martingale. That is the drift "
                    "in the sample. "
                    "An empty box uses the bundled file "
                    "`playground/examples/daily_closes.csv`, eighty synthetic "
                    "Black-Scholes closes, seed 1, volatility 20 percent. Not a market series."
                ),
            ]),
            "Python market": mo.vstack([
                py_train, py_test,
                mo.md(
                    "An empty box uses `playground/templates/merton_jump.py`, a Merton "
                    "jump-diffusion. Copy that file, edit the numbers, and drop it here. "
                    "The file defines `build_market()` and returns a `Market`. "
                    "The upload checks shapes and a martingale condition: on 1 024 paths "
                    "the mean of the discounted terminal spot should lie within 4 "
                    "standard errors of s0. "
                    "A regime-switching volatility is the same kind of file. Draw the "
                    "regime, write down the order of the draws, and implement the same methods."
                ),
                mo.accordion({
                    "Template source": mo.Html(f"<pre>{html.escape(template_text)}</pre>"),
                    "Bundled CSV": mo.Html(f"<pre>{html.escape(example_text)}</pre>"),
                }),
            ]),
        }),
        mo.md("## Hedge"),
        hedger,
        mo.md(
            "Delta is the Black-Scholes delta at the volatility named in the notes. "
            "The Whalley-Wilmott band trades to the nearer edge of an asymptotic "
            "no-trade band around that delta. The fixed band does the same with a "
            "constant half-width. The feed-forward network reads time, log moneyness "
            "and the previous position. The recurrent network reads the same three "
            "numbers and keeps a hidden state along the path. "
            "Networks are small: width 16, depth 2, or a GRU hidden size of 16. "
            "Learning rate 0.005, batch size 500, entropic risk aversion 1."
        ),
        mo.md("## Cost"),
        cost_family, cost_bp, fixed_cash, impact, half_width,
        mo.md(
            "Proportional cost is a fraction of the spot, charged on every change "
            "of holding, including the opening trade and liquidation. "
            "Fixed plus proportional adds a cash amount on each date the position "
            "changes. During training that cash is a smooth tanh. The reported P&L "
            "uses the exact charge. "
            "Quadratic cost is an impact times the squared trade. "
            "The fixed half-width is the constant band, in shares. "
            "The session 00 choice at 50 bp was 0.10."
        ),
        mo.md("## Sample"),
        n_paths, n_steps, epochs, run,
        mo.md(
            "A few thousand paths and about twenty epochs are enough to see the loss "
            "fall on a laptop CPU. The default is a few seconds to a few tens of seconds."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    bergomi_rho, bergomi_xi, block, cost_bp, cost_family, csv_test, csv_train,
    decode_text, demean, epochs, eta, example_closes, fixed_cash, half_width,
    hedger, hurst, impact, kappa, maturity_days, mo, mu, n_paths, n_steps,
    parse_closes, py_test, py_train, rate, rho, s0, sigma, strike, template_text,
    test_card, theta, train_card, v0, xi, MarketUploadError, PlaygroundSpec,
):
    def _file_text(widget):
        if not widget.value:
            return None, None
        name = widget.name() or "upload"
        return name, decode_text(widget.contents(), name)

    def _closes(card, widget):
        if card != "csv":
            return None
        _name, text = _file_text(widget)
        if text is None:
            return example_closes
        return parse_closes(text)

    def _source(card, widget):
        if card != "py":
            return None, "train.py"
        name, text = _file_text(widget)
        if text is None:
            return template_text, "merton_jump.py"
        return text, name

    train_source, train_filename = _source(train_card.value, py_train)
    test_source, test_filename = _source(test_card.value, py_test)
    try:
        spec = PlaygroundSpec(
            train_card=train_card.value,
            test_card=test_card.value,
            s0=float(s0.value),
            strike=float(strike.value),
            maturity_days=float(maturity_days.value),
            r=float(rate.value),
            sigma=float(sigma.value),
            mu=float(mu.value),
            kappa=float(kappa.value),
            theta=float(theta.value),
            xi=float(xi.value),
            rho=float(rho.value),
            v0=float(v0.value),
            hurst=float(hurst.value),
            eta=float(eta.value),
            bergomi_rho=float(bergomi_rho.value),
            bergomi_xi=float(bergomi_xi.value),
            block=int(block.value),
            demean=bool(demean.value),
            hedger=hedger.value,
            cost_family=cost_family.value,
            cost_bp=float(cost_bp.value),
            fixed_cash=float(fixed_cash.value),
            impact=float(impact.value),
            half_width=float(half_width.value),
            m=int(n_paths.value),
            n_steps=int(n_steps.value),
            epochs=int(epochs.value),
            train_closes=_closes(train_card.value, csv_train),
            test_closes=_closes(test_card.value, csv_test),
            train_source=train_source,
            test_source=test_source,
            train_filename=train_filename if train_card.value == "py" else "train.py",
            test_filename=test_filename if test_card.value == "py" else "test.py",
        )
        spec_error = None
    except (TypeError, ValueError, MarketUploadError) as exc:
        spec = PlaygroundSpec(train_source="invalid")
        spec_error = str(exc)
    return spec, spec_error


@app.cell(hide_code=True)
def _(PlaygroundSpec, loss_figure, mo, run_experiment):
    default_seen = []

    def _default_epoch(epoch, value):
        default_seen.append(value)
        mo.output.replace(loss_figure(default_seen, f"Default training, epoch {epoch}"))

    baseline = run_experiment(PlaygroundSpec(), on_epoch=_default_epoch)
    mo.callout(
        mo.md(
            f"Default picture computed in **{baseline.wall_time:.1f} s**. "
            "Train in Black-Scholes, test in Heston, feed-forward network, "
            "50 bp, 2 000 paths, 30 steps, 20 epochs. "
            "Training seed 0. Test seed 12345."
        ),
        kind="neutral",
    )
    return (baseline,)


@app.cell(hide_code=True)
def _(mo):
    get_live, set_live = mo.state(None)
    return get_live, set_live


@app.cell(hide_code=True)
def _(
    MarketUploadError, load_uploaded_market, mo, py_test, py_train, template_text,
    test_card, train_card, validate_market, decode_text,
):
    def _py_source(widget):
        if not widget.value:
            return template_text, "merton_jump.py"
        name = widget.name() or "upload.py"
        return decode_text(widget.contents(), name), name

    messages = []
    upload_error = None
    for _slot, _card, _widget in (
        ("Training", train_card.value, py_train),
        ("Test", test_card.value, py_test),
    ):
        if _card != "py":
            continue
        _source, _filename = _py_source(_widget)
        try:
            _market = load_uploaded_market(_source, _filename)
            messages.append(f"**{_slot}.** {validate_market(_market)}")
        except MarketUploadError as exc:
            upload_error = str(exc)
            messages.append(f"**{_slot}.** {exc}")
    if messages:
        kind = "danger" if upload_error else "success"
        report = mo.callout(mo.md("\n\n".join(messages)), kind=kind)
    else:
        report = mo.md("")
    report
    return (upload_error,)


@app.cell(hide_code=True)
def _(
    loss_figure, mo, run, run_experiment, set_live, spec, spec_error, upload_error,
    PlaygroundError, MarketUploadError,
):
    status = mo.md(
        "The figures use the default picture until you press **Train and score**. "
        "Moving a control does not retrain."
    )
    if run.value:
        if spec_error or upload_error:
            status = mo.callout(mo.md(spec_error or upload_error), kind="danger")
        else:
            live_seen = []

            def _live_epoch(epoch, value):
                live_seen.append(value)
                mo.output.replace(
                    loss_figure(live_seen, f"Training on the current controls, epoch {epoch}")
                )

            try:
                trained = run_experiment(spec, on_epoch=_live_epoch)
                set_live(trained)
                if trained.history:
                    finished = (
                        f"Finished in **{trained.wall_time:.1f} s**. {trained.label}. "
                        f"Training objective from {trained.history[0]:.3f} to "
                        f"{trained.history[-1]:.3f}."
                    )
                else:
                    finished = (
                        f"Finished in **{trained.wall_time:.1f} s**. {trained.label}. "
                        "This hedge is a formula, so there is no training loss."
                    )
                status = mo.callout(mo.md(finished), kind="success")
            except (PlaygroundError, MarketUploadError, ValueError) as exc:
                status = mo.callout(mo.md(str(exc)), kind="danger")
    status
    return


@app.cell(hide_code=True)
def _(
    baseline, band_figure, get_live, histogram_figure, loss_figure, mo,
    path_figure, spec, spec_id, stats_table,
):
    result = baseline if get_live() is None else get_live()
    if spec_id(spec) == result.spec_id:
        fresh = "The figures match the controls."
    else:
        fresh = (
            f"The controls have moved. The figures are still: {result.label}. "
            "Press Train and score to draw them again."
        )
    if result.same_market:
        miss = (
            f"Train and test are both {result.train_name}. "
            "The test row is a fresh sample with seed 12345, not a new model."
        )
    else:
        miss = (
            f"Misspecification: the hedge is trained in {result.train_name} and "
            f"scored in {result.test_name}. Misspecification means the two markets "
            "are different models. The premium in the P&L is the training price. "
            "The test paths are drawn from the other model, so a hedge that fitted "
            "the training sample can lose money here."
        )
    if result.history:
        curve = loss_figure(
            result.history,
            f"Entropic risk on the training paths after each epoch. {result.hedger_name}.",
        )
    else:
        curve = mo.md(
            "Delta, the Whalley-Wilmott band and the fixed band are formulas. "
            "They have no training loss. Choose a network to see the curve."
        )
    notes = "\n".join(f"- {note}" for note in result.notes)
    mo.vstack([
        mo.callout(mo.md(fresh), kind="neutral"),
        mo.callout(mo.md(miss), kind="warn" if not result.same_market else "neutral"),
        mo.md("## Loss"),
        mo.md(
            "Entropic risk is the cash you would add so that the expected "
            "exponential utility of the P&L equals the utility of zero. "
            "Larger is worse. The risk aversion lambda is 1. "
            "The curve is the full training sample after each epoch, with the exact cost."
        ),
        curve,
        mo.md("## P&L"),
        histogram_figure(result),
        mo.md(stats_table(result)),
        mo.md(
            "CVaR at 95 percent is the average loss on the worst 5 percent of paths. "
            "The standard errors are sampling error on this batch, not across training seeds."
        ),
        mo.md("## One path"),
        path_figure(result),
        mo.md(
            "The solid red shares are the hedge named above. The dashed blue shares "
            "are delta. Both are marked against the right axis."
        ),
        mo.md("## No-trade band"),
        band_figure(result),
        mo.md(
            "A no-trade band is a range around delta. Inside it the hedge keeps the "
            "shares it already holds. The two solid curves are the hedge when it "
            "already holds 0 shares and when it already holds 0.5 shares. Where they "
            "separate, the previous holding still matters. "
            "The Whalley-Wilmott band is an asymptotic half-width for a proportional "
            "cost. It grows like the cube root of the cost times gamma squared. "
            "The shade uses that Black-Scholes formula and the test volatility "
            f"({result.sigma_test:.3f}, {result.sigma_note_test}). "
            "On a recurrent network the curves are the empty-memory answer. "
            "The sample path above carries the memory from date to date."
        ),
        mo.md("## Notes for this run\n" + notes),
    ])
    return


if __name__ == "__main__":
    app.run()


