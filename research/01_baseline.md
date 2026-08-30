# 01 — Baseline

**Status: AWAITING DATA.** Pre-registered before any result was observed.

## Hypothesis

**H0 (null):** A model trained on the feature set in `nqlab.features.technical`
produces expected value per trade indistinguishable from zero after costs, on
out-of-sample walk-forward folds.

**H1:** It produces EV per trade > 0 after costs, with a lower 95% bootstrap
confidence bound above zero.

This report exists to make it hard to skip. Most published strategy work never
states a null, so it is never rejected — it is simply not mentioned once the
equity curve looks good.

## Baselines to beat

A model is only interesting relative to something. Four references:

| Baseline | Why |
|---|---|
| **Always flat** | EV exactly 0, minus nothing. Any strategy that loses to this should not exist. |
| **Always long** | NQ has risen over most historical windows. A "predictive" model that cannot beat buy-and-hold has predicted nothing. |
| **Random entries** | Same trade count and holding period, random direction. Isolates whether the *signal* matters or just the exposure and the barrier geometry. |
| **Single-feature rule** | Momentum sign alone. If a 15-feature GBM cannot beat one comparison, the complexity is decoration. |

The random-entry baseline is matched on trade count, because a strategy that
trades 10× more often is not comparable on total P&L.

## Method

- **Data:** NQ continuous front month, 5-minute bars, Panama-adjusted with a
  causal volume-based roll. Range recorded at run time.
- **Labels:** triple-barrier, 1 ATR stop, 2R target, 48-bar horizon, costs on.
- **Split:** anchored walk-forward, embargo = 48 bars (the label horizon).
  Hyperparameters selected on a validation slice at the end of each training
  fold; test folds touched once.
- **Model:** L2 logistic regression on standardised features. Deliberately the
  simplest thing that could work — starting with a boosted ensemble makes it
  impossible to tell later whether complexity bought anything.
- **Weights:** label uniqueness, to stop overlapping labels inflating the
  effective sample size.

## Metrics

Reported per fold **and** aggregated, because a strategy that works in one fold
and fails in four is not a strategy.

Statistical: PR-AUC, Brier score, log loss, reliability curve.
Economic: EV per trade in R, profit factor, max drawdown in R, trade count,
win rate, and the breakeven win rate implied by the target geometry.

**Decision metric:** out-of-sample EV per trade in R after costs, with a
bootstrap 95% CI over folds.

## Falsification criteria — stated in advance

The result is **negative**, and will be published as negative, if any of:

- The lower 95% bootstrap bound on EV per trade is ≤ 0.
- EV does not exceed the random-entry baseline at matched trade count.
- Fewer than 3 of 5 folds are individually positive (an edge concentrated in one
  period is a period, not an edge).
- Total out-of-sample trades < 200 (too few to distinguish from noise).

## Results

`AWAITING DATA`

## Conclusion

`AWAITING DATA`
