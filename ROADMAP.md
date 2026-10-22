# Roadmap

The session ladder. Levels are defined in `CLAUDE.md`: 0 play, 1 reproduce,
2 implement, 3 extend. A box is ticked when Corey has worked through the session.

- [ ] **00 Delta hedging loses money when trading costs money** (level 0).
  `sessions/00_cost_of_hedging.py`. Black-Scholes market, short call, delta hedge against
  a small network trained to minimise CVaR of P&L. The no-trade band appears when
  cost > 0.
- [ ] **01 Reproduce Buehler et al. 2019** (level 1). A Heston market, hedging with the
  underlying and a variance swap. Reproduce one figure of the paper side by side with
  the original.
- [ ] **02 The Whalley-Wilmott band** (level 2). Implement the asymptotic band of Whalley
  and Wilmott (1997) analytically and put it on the same axes as the learned band,
  across spots, times and costs.
- [ ] **03 Recurrent policy and path-dependent features** (level 2). Replace the
  feature-based network by a recurrent one and add path features. Check that nothing
  changes in a Markov market and that something does in a path-dependent one.
- [ ] **04 Deep hedging under rough Bergomi** (level 1). The setup of Horvath, Teichmann
  and Zuric (2021). The rough Bergomi simulator comes from the sibling repository
  [rough-volatility](https://github.com/coreyjpritchard/rough-volatility), package `roughvol`.
- [ ] **05 A level 3 question** (level 3). Proposed: does the width of the learned
  no-trade band scale with the Hurst exponent H under rough Bergomi, and does a
  feature-based network lose more to a recurrent one as H falls?

## Environment

The market steps under the sessions above. A box is ticked when that market
runs on the shared interface, which is separate from working through the session.

- [ ] **Shared market environment, Black-Scholes.** One interface for instruments,
  features, payoff, price and a benchmark hedge. Proportional costs plug in beside
  the market. Session 00 runs on it.
- [ ] **Heston on the same interface.** Features `(spot, variance)`. A variance swap
  can be switched on as a second hedging instrument. This is the market for session 01.
- [ ] **Pluggable transaction costs.** Cost models are part of the environment and
  independent of the volatility model. The proportional cost is the first; the
  others are not written yet.
- [ ] **Rough Bergomi on the same interface.** Features carry volatility memory. The
  simulator comes from the sibling repository
  [rough-volatility](https://github.com/coreyjpritchard/rough-volatility), package
  `roughvol`. This is the market for session 04.
