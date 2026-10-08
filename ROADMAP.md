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
