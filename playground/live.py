import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Live hedging")


@app.cell
def _():
    import io

    import marimo as mo
    from matplotlib import pyplot as plt

    from deephedge.live import (
        describe,
        describe_batch,
        draw_batch,
        draw_live,
        market_by_name,
        score_books,
        simulate_live,
    )

    def png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    return (
        png,
        describe,
        describe_batch,
        draw_batch,
        draw_live,
        market_by_name,
        mo,
        score_books,
        simulate_live,
    )


@app.cell
def _(mo):
    get_frame, set_frame = mo.state(0)
    get_seed, set_seed = mo.state(7)
    get_ready, set_ready = mo.state(False)
    get_batch, set_batch = mo.state(None)
    return (
        get_batch,
        get_frame,
        get_ready,
        get_seed,
        set_batch,
        set_frame,
        set_ready,
        set_seed,
    )


@app.cell
def _(mo, set_batch, set_frame):
    def _reset(_value):
        set_frame(0)
        set_batch(None)

    market = mo.ui.dropdown(
        options=["Black-Scholes", "Heston"],
        value="Black-Scholes",
        label="Market",
        on_change=_reset,
    )
    steps = mo.ui.slider(
        10, 60, step=1, value=30, show_value=True, label="Steps", on_change=_reset,
    )
    cost = mo.ui.slider(
        0, 200, step=5, value=50, show_value=True, label="Cost, basis points",
        on_change=_reset,
    )
    width = mo.ui.slider(
        0.0, 0.40, step=0.01, value=0.10, show_value=True,
        label="Band half-width, shares", on_change=_reset,
    )
    return cost, market, steps, width


@app.cell
def _(mo, set_batch, set_frame, set_seed, steps):
    def _step(_value):
        last = int(steps.value)
        set_frame(lambda i: min(int(i) + 1, last))
        return 0

    def _restart(_value):
        set_frame(0)
        return 0

    def _new_path(_value):
        set_seed(lambda seed: int(seed) + 1)
        set_frame(0)
        set_batch(None)
        return 0

    play = mo.ui.checkbox(label="Play")
    speed = mo.ui.refresh(
        options=["0.1s", "0.25s", "0.5s", "1s", "2s"],
        default_interval="0.5s",
        label="Speed",
    )
    step = mo.ui.button(label="Step", on_click=_step)
    restart = mo.ui.button(label="Restart", on_click=_restart)
    new_path = mo.ui.button(label="New path", on_click=_new_path)
    return new_path, play, restart, speed, step


@app.cell
def _(get_seed, mo, set_batch, set_frame, set_seed):
    def _on_seed(value):
        set_seed(int(value))
        set_frame(0)
        set_batch(None)

    seed = mo.ui.number(
        0, 99_999, step=1, value=int(get_seed()), label="Seed", on_change=_on_seed,
    )
    return (seed,)


@app.cell
def _(
    cost, market, mo, new_path, play, restart, seed, speed, step, steps, width,
):
    mo.vstack([
        mo.hstack([market, steps, cost, width, seed], justify="start", gap=1),
        mo.hstack(
            [play, speed, step, restart, new_path],
            justify="start",
            gap=1,
            align="center",
        ),
        mo.md(
            "Play advances one date on the speed clock. A shorter interval is faster. "
            "Step moves one date. New path adds one to the seed. "
            "The vertical scale is the whole path, so the axes do not jump. "
            "The lines stop at the current date."
        ),
    ])
    return


@app.cell
def _(get_frame, mo, set_frame, steps):
    last = int(steps.value)
    date = mo.ui.slider(
        0,
        last,
        step=1,
        value=min(int(get_frame()), last),
        show_value=True,
        include_input=True,
        label="Date",
        on_change=lambda value: set_frame(int(value)),
    )
    date
    return (date,)


@app.cell
def _(get_frame, play, set_frame, speed, steps):
    _tick = speed.value
    if play.value and int(get_frame()) < int(steps.value):
        set_frame(lambda i: min(int(i) + 1, int(steps.value)))
    return


@app.cell
def _(get_frame, get_ready, set_ready, steps):
    reached = int(get_frame()) >= int(steps.value)
    if reached != bool(get_ready()):
        set_ready(reached)
    return


@app.cell
def _(cost, market, market_by_name, seed, simulate_live, steps, width):
    book = simulate_live(
        market_by_name(str(market.value)),
        n=int(steps.value),
        seed=int(seed.value),
        half_width=float(width.value),
        cost=float(cost.value) / 1e4,
    )
    return (book,)


@app.cell
def _(book, date, describe, draw_live, mo, png):
    index = min(max(int(date.value), 0), book.n)
    fig = draw_live(book, index)
    mo.vstack([
        mo.image(png(fig), alt="Spot, band, cash and hedge result up to this date"),
        mo.callout(mo.md(describe(book, index)), kind="neutral"),
    ])
    return


@app.cell
def _(get_ready, mo):
    go = mo.ui.run_button(
        label="Score 2,000 paths",
        disabled=not bool(get_ready()),
        tooltip="Same two hedgers, fresh paths. A few seconds on Heston.",
    )
    go
    return (go,)


@app.cell
def _(book, go, market_by_name, score_books, set_batch):
    if go.value:
        set_batch(score_books(
            market_by_name(book.market_name),
            n=book.n,
            m=2_000,
            seed=book.seed + 10_007,
            half_width=book.half_width,
            cost=book.cost,
            token=book.token,
        ))
    return


@app.cell
def _(book, describe_batch, draw_batch, get_batch, mo, png):
    score = get_batch()
    if score is None or score.token != book.token:
        picture = mo.md(
            "When this path has finished, **Score 2,000 paths** draws a fresh sample, "
            "seed plus 10007, and marks this path on the histogram. "
            "On Heston the cosine delta takes a few seconds."
        )
    else:
        batch_fig = draw_batch(book, score)
        picture = mo.vstack([
            mo.image(png(batch_fig), alt="Terminal result of the band and of delta hedging"),
            mo.md(describe_batch(score)),
        ])
    picture
    return


@app.cell
def _(book, mo):
    mo.md(
        f"""
        {book.delta_note}

        The holding is clipped to a fixed band around delta. Inside the band it
        stays where it is. When delta moves past it, the holding is pushed to
        the nearer edge, and that date is marked. Delta hedging is the same
        rule with half-width zero, on this path, at this cost.

        Cash is time-0 money. It starts as the premium, then pays for shares
        and for a proportional cost on the discounted spot. The hedge P/L is
        that cash, plus the shares, minus the model price of the call. At
        maturity the shares are sold and the call is paid. That last number
        is the terminal P/L, the same one the histogram counts. A dot is a
        rebalance, including the open and the close.

        The lessons that build a network up to this picture, one idea at a
        time, are in `LEARN.md`. Nothing in that ladder is built yet.
        """
    )
    return


if __name__ == "__main__":
    app.run()
