# NQLab

Research and decision-support tooling for NQ (E-mini Nasdaq-100) futures.

> ### This is not a trading bot and makes no claim of profitability
>
> NQLab does not predict prices. It estimates probabilities, states how well
> calibrated they are, and refuses to produce a setup when the evidence does not
> support one. **No backtest result is presented in this repository, because no
> market data was available in the environment where it was built.** Every
> research report is a pre-registered protocol with its results section empty.
>
> Nothing here should be used to make financial decisions.

**Stack:** Python 3.11 · pandas · NumPy · scikit-learn · pytest · Decimal arithmetic for all money

---

## What this is, and what it is not

Most "AI trading" repositories fail for reasons that have nothing to do with the
model: information leaks from the future into the features, validation shuffles
an autocorrelated series, costs are applied last or not at all, and a stated
probability is never checked against how often the thing actually happens.

NQLab is built the other way round. **The prediction model is the least
important component and is deliberately built last.** What exists so far is the
machinery that would make a model's results believable — and, critically, the
tests that prove that machinery works.

### Current status: honest inventory

| Component | State | Verified how |
|---|---|---|
| Contract specs, tick/point arithmetic | ✅ Done | Decimal throughout; tick rounding tested |
| Session and trading-date classification | ✅ Done | Tested incl. the 18:00 ET session roll |
| Provider abstraction (CSV / yfinance / replay) | ✅ Done | Replay provider raises on look-ahead |
| Feature framework + 15 features | ✅ Done | Every feature causality-tested |
| **Leakage detection suite** | ✅ Done | **Proven to catch 6 classes of real leak** |
| Triple-barrier labelling | ✅ Done | Hand-built fixtures with known answers |
| Walk-forward splitter with embargo | ✅ Done | 11 property tests |
| Risk engine / position sizing | ✅ Done | Swept invariants over 100s of cases |
| Regime detection | ⬜ Not built | — |
| Models, calibration, backtest execution | ⬜ Not built | — |
| FastAPI service, Next.js dashboard, agent | ⬜ Not built | — |
| **Any empirical result** | ⬜ **None** | **No data access — see below** |

**124 tests pass.** None of them assert anything about the market; they assert
that the engine is correct.

---

## The central claim: leakage cannot pass silently

Every out-of-sample number a system like this reports is worthless if a feature
reads the future. So that guarantee is mechanical rather than a promise.

### The truncation property

For every feature `f`, dataset `D`, and cut point `t`:

```
f(D[0:t]).iloc[t-1]  ==  f(D).iloc[t-1]
```

A feature computed on history truncated at `t` must give the **identical** value
at `t-1` as one computed on the full series. Any peek at the future — a centred
window, a negative shift, a full-series `fit`, a rank against the whole sample —
breaks this equality.

It runs against **every feature in the registry**, at **every cut point past
warm-up**, so a newly registered feature is covered automatically and no
reviewer has to remember to add a test.

### The detector is itself tested

A leakage suite that never fires proves nothing. `tests/test_leakage_detector_works.py`
constructs six features that leak the way real code leaks, and asserts the
detector catches each:

| Deliberate leak | Caught by |
|---|---|
| `shift(-1)` — reads the next bar | truncation |
| `rolling(center=True)` — half the window is future | truncation, append |
| Full-series z-score | truncation, append |
| Full-series percentile rank | truncation, append |
| Distance from the series maximum | truncation |
| `bfill()` over periodic gaps | truncation |

**This test found a real hole during development.** The truncation check
originally sampled four fixed cut points, and `bfill` slipped between them
because none happened to land on a gap row. The sweep is now dense. That hole
existed, was found by the meta-test, and is why the meta-test is in the
repository rather than being described as unnecessary.

A seventh case asserts the converse: a genuinely causal feature must *not* trip
the detector, so a detector that returned "leak" unconditionally would fail too.

---

## What it predicts

Not "will the next candle close higher" — that target is close to unpredictable
at any tradeable horizon, and optimising for it yields a model that is right 51%
of the time and loses money after costs.

The target is a **cost-aware, path-dependent triple barrier**: from a candidate
entry, does price reach `+kR` **before** `−1R`, within a bounded holding period?
Outcomes are `WIN` / `LOSS` / `TIMEOUT`.

Three decisions inside it are worth reading the code for:

**Both barriers inside one bar resolves as a LOSS.** OHLC data cannot reveal
intrabar order. Assuming the favourable order is the most common way a backtest
gets inflated without anyone writing a dishonest line.

**Costs make winning harder, never easier.** The target must clear the round-trip
cost; the stop does not move, because a resting order does not move because
commission exists. An earlier implementation pulled the target in *and* pushed
the stop out — both easier — which would have inflated every win rate the system
reported. A swept test now asserts win count is monotonically non-increasing in
cost.

**The unresolved tail stays NaN.** Bars without a full forward window are not
labelled `TIMEOUT`; recording them as non-events biases the end of every sample.

**Label overlap is measured.** A label spanning `H` bars shares outcome
information with the next `H` labels. Uniqueness is computed and used as a
sample weight, and the walk-forward embargo is set to `H`.

---

## Risk engine

Pure `Decimal` arithmetic — no floats anywhere money is involved, because a
sizing routine that accumulates float error reports a P&L that will not
reconcile against a hand-tallied trade list.

It sizes **down, never up**, and it declines when it must:

```python
config = RiskConfig(account_size=D("50000"), risk_per_trade=D("0.005"))
size_position(entry=D("18000"), stop=D("17975"), spec=NQ, config=config)
# → 0 contracts
#   "Stop of 25.00 points risks $514.00/contract, which exceeds the $250.00
#    budget. Widen the account, tighten the stop, or use a smaller contract (MNQ)."
```

A $50k account at 0.5% risk genuinely cannot trade one NQ contract on a
25-point stop. Rounding up to 1 — which naive sizing does — would risk 2× the
stated budget on every such trade. Tests sweep stop distances and assert the
budget is never exceeded, that sizing is monotone in stop distance, and that
costs are charged against the budget rather than ignored.

---

## Bring your own data

The environment this was built in had **no network access to any market data
provider**, which is why there are no results. Everything is ready for data:

```bash
pip install -e ".[dev,data]"

# Option 1 — your own files
mkdir -p data/NQ && cp your_bars.csv data/NQ/5min.csv
# columns: timestamp,open,high,low,close,volume

python -c "
from nqlab.data.providers import CsvProvider, BarRequest
bars = CsvProvider('./data').get_bars(BarRequest('NQ', '5min'))
print(bars.tail())
"
```

`validate_bars` rejects rather than repairs: unsorted or duplicate timestamps,
naive timezones, non-positive prices, negative volume, and bars where
`high < max(open, close, low)`. Silent repair of bad vendor data is how a corrupt
series reaches a model unnoticed.

**On yfinance:** supported, and adequate for development only. Intraday history
is capped near 60 days, futures volume is unreliable, and the continuous `NQ=F`
series is stitched by an undocumented method — so it is not a basis for a
published result.

---

## Validation protocol

Anchored walk-forward with an embargo, fixed before any result exists:

```
fold 1  ├── train ──┤─val─┤▒embargo▒├─ test ─┤
fold 2  ├──── train ────┤─val─┤▒embargo▒├─ test ─┤
```

- Chronological only. No shuffling, no k-fold, ever.
- Hyperparameters chosen on a validation slice at the **end of training**; test
  folds are touched once.
- The embargo is at least the label horizon, or the last training labels resolve
  using test-period bars.
- Costs always on. No result is reported gross.
- Metrics per fold **and** aggregated — an edge in one fold out of five is a
  period, not an edge.

`assert_no_overlap` and `embargo_is_sufficient` are exported so a research script
can assert its own splits are sound rather than assuming it.

---

## Testing

```bash
pytest                                  # 124 tests
pytest tests/test_leakage.py -v         # causality of every feature
pytest tests/test_leakage_detector_works.py -v -s   # the detector's own coverage
```

| Suite | Tests | What it protects |
|---|---|---|
| `test_leakage` | 62 | No feature reads the future, at any cut point |
| `test_leakage_detector_works` | 13 | The detector catches 6 real leak classes |
| `test_risk` | 21 | Budget never exceeded; sizing monotone; costs charged |
| `test_labels` | 17 | Barrier order, cost direction, unresolved tail, overlap |
| `test_splitter` | 11 | No fold overlap; embargo; train never touches test |

CI runs lint, format, `mypy --strict` and tests on 3.11 and 3.12, with the
**leakage gate as a separate job** so a causality failure is legible as its own
red check rather than one line inside a larger run.

Synthetic bars are used to exercise engine correctness — causality, fills,
arithmetic. **No result derived from synthetic data is reported as a finding
about the market**, and the fixtures say so.

---

## What is not built

Stated plainly, because a status table that only lists what exists is marketing:

- **No models.** No logistic regression, no GBM, no sequence model. The
  experiment framework is designed (`docs/ARCHITECTURE.md` §6) and unwritten.
- **No backtest execution engine.** The splitter exists; the event loop that
  walks bars, fills orders and tracks equity does not.
- **No regime detection, no calibration, no agent, no API, no dashboard.**
- **No empirical result of any kind.**

The order is deliberate: the components that make results *trustworthy* are
built and tested first, so that when a model is added its numbers mean
something.

## Roadmap

1. Backtest execution engine — fills, slippage, partials — tested against
   hand-built fixtures with known P&L before any strategy runs through it.
2. Baseline models per `research/01_baseline.md`, including the four baselines
   any model must beat.
3. Probability calibration (isotonic / Platt) with reliability curves, and the
   check that a stated 68% occurs ~68% of the time.
4. Regime detection, tested for whether it improves EV or merely splits the sample.
5. No-trade engine, measured on whether it improves risk-adjusted performance.
6. FastAPI service and Next.js dashboard.
7. Agent explanation layer — reading structured output, with **no code path back
   into the risk engine**.

## Licence

MIT. Not investment advice. No warranty of any kind.
