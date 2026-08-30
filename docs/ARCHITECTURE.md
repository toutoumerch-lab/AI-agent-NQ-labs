# NQLab — Architecture and Research Design

> **This is a research and decision-support system. It does not predict prices,
> and it does not guarantee profitable trading. Every probability it emits is an
> estimate with error bars, and a large part of the engineering below exists to
> stop that estimate from being quietly wrong.**

---

## 0. The honest starting position

Most published "AI trading" work fails for one of four reasons, none of which is
model architecture:

1. **Leakage.** A feature or a normalisation touches information from after the
   decision point. The backtest looks excellent and the live system does not.
2. **Wrong validation.** Random train/test splits on autocorrelated series, or
   tuning against the same period used to report results.
3. **Costs applied last, if at all.** An edge of 0.3 ticks per trade evaporates
   under a 1-tick spread plus commission.
4. **Uncalibrated confidence.** "68%" that is not 68% is worse than no number,
   because it is sized on.

NQLab is organised so that each of those is a component with tests, not an
assumption. **The prediction model is the least important part of this system**,
and it is deliberately built last.

---

## 1. Instrument: NQ (E-mini Nasdaq-100 futures)

Contract facts that the code must encode rather than assume:

| Property | Value | Why it matters in code |
|---|---|---|
| Exchange | CME (Globex) | Session calendar, holidays |
| Multiplier | **$20 × index** | P&L conversion; a 1-point move is $20/contract |
| Tick size | **0.25 index points** | All prices must be tick-aligned; stops and targets must round to a valid tick |
| Tick value | **$5.00** | 0.25 × $20 |
| Contract months | Mar, Jun, Sep, Dec (H, M, U, Z) | Rollover logic |
| Expiry | Third Friday of contract month | Roll window |
| RTH | 09:30–16:00 America/New_York | Session features, opening range |
| Globex | 18:00 prior day – 17:00, Sun–Fri ET | Overnight range, gap definition |
| Daily halt | 17:00–18:00 ET | Legitimate gap in the bar series — not missing data |

**Micro contract (MNQ)** is 1/10 the size ($2 multiplier, $0.50 tick value) and is
supported as a separate spec so position sizing can use it for small accounts.

### Rollover

Front-month data must be stitched, and *how* it is stitched changes every
historical return. Three options, and the choice is recorded in metadata:

| Method | Effect | Use |
|---|---|---|
| `NONE` | Raw front month, price jumps at roll | Never for modelling |
| `PANAMA` (back-adjust by difference) | Continuous, returns preserved near the roll; **historical prices are not real prices** | Default for features/labels |
| `RATIO` | Multiplicative adjustment | Long histories where % scale matters |

Roll trigger is **volume-based** (roll when the back month's volume exceeds the
front's for N consecutive sessions), not calendar-based, because the actual
liquidity migration leads the calendar by days and varies by quarter.

**This is a leakage surface.** A back-adjusted series computed over the full
history uses future roll gaps to adjust past prices. NQLab therefore requires
the adjustment to be computed **causally** — the adjustment applied to bar `t`
uses only rolls at or before `t` — and a test enforces it.

---

## 2. What the system actually predicts

Not "the next candle." The naive target is close to unpredictable at any horizon
that matters, and optimising for it produces a model that is right 51% of the
time and loses money after costs.

The primary target is a **path-dependent, cost-aware, triple-barrier label**:

> From a candidate entry at bar `t`, with stop distance `s` and target distance
> `k·s` (both volatility-scaled), does price touch `+k·s` **before** `−s`, within
> a maximum holding time `H`?

Outcomes: `WIN` / `LOSS` / `TIMEOUT`. This is the López de Prado triple-barrier
method, and it is used because it is the only labelling that answers the
question a trader actually faces. It has three properties the naive target lacks:

- It is **path-dependent** — order of arrival matters, so it cannot be satisfied
  by a forecast that is directionally right but drawn down through the stop first.
- It is **cost-aware** — barriers are set net of spread, slippage and commission,
  so a "win" is a win after costs.
- It is **horizon-bounded** — `TIMEOUT` is a real, distinct outcome rather than
  being silently forced into win/loss.

Secondary targets are implemented for comparison, because assuming the primary
is best without testing is exactly the error this project is meant to avoid:

| Target | Type | Why test it |
|---|---|---|
| Triple-barrier | 3-class | Primary; directly tradeable |
| Directional sign over `H` | binary | Baseline; is path-dependence worth it? |
| MFE / MAE | regression | Sizing and target placement |
| Forward return | regression | Does a continuous target beat a discretised one? |

**Selection rule, fixed before any results are seen:** the target that yields the
highest out-of-sample expected value per trade after costs, on the walk-forward
protocol in §5, with ties broken toward fewer parameters.

### Sample weighting

Triple-barrier labels **overlap** — a label from bar `t` spanning `H` bars shares
outcome information with labels from `t+1…t+H`. Treating them as independent
inflates effective sample size and every significance estimate built on it.
NQLab computes **label uniqueness** (average fraction of the label's span not
shared with other labels) and uses it as a sample weight, and the walk-forward
splitter applies an **embargo** of `H` bars between train and test.

---

## 3. Component architecture

```
┌───────────────────────────────────────────────────────────────┐
│ providers/     MarketDataProvider (interface)                  │
│                ├─ CsvProvider          (files you supply)      │
│                ├─ YFinanceProvider     (needs network)         │
│                └─ ReplayProvider       (bar-at-a-time, for the │
│                                         causality harness)     │
└─────────────────────────────┬─────────────────────────────────┘
                              ↓  raw bars
┌───────────────────────────────────────────────────────────────┐
│ data/          Contract specs · session calendar · timezone    │
│                normalisation · gap and duplicate handling ·    │
│                causal rollover stitching                       │
└─────────────────────────────┬─────────────────────────────────┘
                              ↓  clean, session-tagged bars
┌───────────────────────────────────────────────────────────────┐
│ features/      Pure functions, one per feature family.         │
│                EVERY feature is causal by construction and     │
│                property-tested against future bars.            │
└─────────────────────────────┬─────────────────────────────────┘
                              ↓  feature matrix + labels
┌──────────────┬──────────────┬──────────────┬──────────────────┐
│ regime/      │ models/      │ vol/         │ calibration/     │
│ classify     │ entry prob.  │ forecast σ   │ isotonic/Platt   │
└──────────────┴──────┬───────┴──────────────┴──────────────────┘
                      ↓  calibrated probabilities
┌───────────────────────────────────────────────────────────────┐
│ risk/          Position sizing · stop/target · R · limits      │
│ entry/         Threshold gates · no-trade filters              │
└─────────────────────────────┬─────────────────────────────────┘
                              ↓  structured Setup (JSON)
┌───────────────────────────────────────────────────────────────┐
│ agent/         LLM reads the structured output and explains.   │
│                It has NO authority to create or alter a setup. │
└─────────────────────────────┬─────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│ api/ (FastAPI)  ·  dashboard/ (Next.js)                        │
└───────────────────────────────────────────────────────────────┘
```

**The dependency rule:** `features/`, `risk/`, `backtest/` import nothing from
`api/`, `agent/`, or any provider. They are pure enough to test exhaustively,
which is the entire point.

**The agent's authority is bounded by construction.** It receives a completed
`Setup` object and produces prose. It cannot change a probability, move a stop,
or turn a `NO_TRADE` into a trade — there is no code path from the agent back
into the risk engine. This is enforced by the type signature, not by a prompt
instruction.

---

## 4. Leakage prevention — the central engineering claim

Every other component is worthless if this one is wrong. Four mechanisms:

### 4.1 Causal-by-construction features

Every feature function takes a `DataFrame` and returns a series aligned to it,
under one rule: **the value at index `t` may depend only on rows `0…t`**. No
centred windows, no `shift(-n)`, no full-series `fit`.

### 4.2 The truncation property test

The mechanism that makes the claim checkable rather than aspirational:

> For every feature `f`, every dataset `D`, and every cut point `t`:
> `f(D[0:t])[t-1] == f(D)[t-1]`

Computing a feature on data truncated at `t` must give the identical value at
`t-1` as computing it on the full history. Any use of future information breaks
this. It is run over **every registered feature** with `hypothesis`-generated
data and random cut points, so a new feature is covered the moment it is
registered — there is no way to add a leaky feature and have the suite stay green.

### 4.3 Fit/transform separation

Scalers, encoders and calibrators are `fit` on training folds only and applied
to validation and test. A full-series `StandardScaler` leaks the test-period mean
and standard deviation into training. The pipeline API makes the leaking version
inexpressible: transforms take a fitted state, and fitting requires a fold.

### 4.4 Embargo

Between the end of a training fold and the start of a test fold, `H` bars are
discarded, where `H` is the maximum label horizon. Without it, the last training
labels resolve using bars that are inside the test period.

---

## 5. Validation protocol — fixed in advance

**Anchored walk-forward**, with an embargo:

```
fold 1  ├── train ──┤▒embargo▒├─ test ─┤
fold 2  ├──── train ────┤▒embargo▒├─ test ─┤
fold 3  ├────── train ──────┤▒embargo▒├─ test ─┤
```

Rules, committed before results exist:

- **Chronological only.** No shuffling, no k-fold, ever.
- **Test folds are touched once.** Hyperparameters are chosen on a validation
  slice carved from the *end of the training fold*, never on test.
- **Costs always on.** No result is reported gross.
- Metrics are aggregated across folds and reported **per fold as well**, because
  a strategy that works in one regime and fails in three is not a strategy.

### Metrics

Statistical: PR-AUC (the classes are imbalanced, so ROC-AUC flatters),
Brier score, reliability curve, log loss.

Economic (these decide): expected value per trade in R, profit factor, max
drawdown in R, trade count, win rate, average win/loss, **performance by regime
and by session**.

**Trade count is a first-class metric.** A model with a superb edge over eleven
trades has told you nothing.

---

## 6. Pre-registered experiments

Because results cannot be produced without data, `research/` contains
**pre-registered protocols** — hypothesis, method, metrics, and the decision rule
— written before any result is seen. This is deliberate: a hypothesis registered
after seeing results is not a hypothesis. Each report has a `Results` section
marked `AWAITING DATA` and is filled in by running the documented command.

| Report | Question |
|---|---|
| `01_baseline.md` | Does anything beat "always flat" and "always long" after costs? |
| `02_feature_analysis.md` | Which feature families carry signal, and are they stable across folds? |
| `03_regime_analysis.md` | Does conditioning on regime improve out-of-sample EV, or just split the sample? |
| `04_model_comparison.md` | Logistic → GBM → sequence models. Does complexity pay? |
| `05_walk_forward.md` | Is performance stable across folds, or driven by one period? |
| `06_calibration.md` | Does a stated 68% occur 68% of the time? |
| `07_ablation.md` | Which components actually contribute? |
| `08_transaction_costs.md` | At what cost level does the edge disappear? |
| `09_final_model.md` | Final selection, with the limitations that survived. |

**Falsification criteria are stated up front.** If the baseline in `01` is not
beaten after costs, that is the finding, and it gets published in the README
rather than buried.

---

## 7. What this system will not do

- It will not place live orders. Analysis and paper trading only.
- It will not state a probability without a sample size and a calibration curve.
- It will not report a backtest without its cost assumptions printed alongside.
- It will not let the language model invent, alter, or override a number.
- It will not claim an edge it has not measured out of sample.
