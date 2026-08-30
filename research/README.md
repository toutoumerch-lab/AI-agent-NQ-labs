# Research protocols

Each report here is **pre-registered**: hypothesis, method, metrics and the
decision rule are written *before* any result is seen, and the `Results`
section is filled in only by running the documented command.

This is deliberate. A hypothesis formed after looking at results is not a
hypothesis, and a threshold chosen after seeing which value performs best is a
fitted parameter masquerading as a design decision. Pre-registration is the
cheapest available defence against both.

**Status: all reports are `AWAITING DATA`.** The environment this repository was
developed in has no access to market data providers, so no empirical result has
been produced. Nothing here is filled in with plausible-looking numbers — that
would be fabrication, and it is the single fastest way to make a quantitative
project worthless.

| Report | Question | Status |
|---|---|---|
| `01_baseline.md` | Does anything beat "always flat" and "always long" after costs? | AWAITING DATA |
| `02_feature_analysis.md` | Which feature families carry signal, and is it stable across folds? | AWAITING DATA |
| `03_regime_analysis.md` | Does conditioning on regime improve out-of-sample EV? | AWAITING DATA |
| `04_model_comparison.md` | Logistic → GBM → sequence models. Does complexity pay? | AWAITING DATA |
| `05_walk_forward.md` | Is performance stable across folds, or one lucky period? | AWAITING DATA |
| `06_calibration.md` | Does a stated 68% happen 68% of the time? | AWAITING DATA |
| `07_ablation.md` | Which components actually contribute? | AWAITING DATA |
| `08_transaction_costs.md` | At what cost level does any edge disappear? | AWAITING DATA |
| `09_final_model.md` | Final selection and surviving limitations. | AWAITING DATA |

## Reproducing

Supply data (see the README's *Bring your own data* section) and run:

```bash
python -m nqlab.research run 01_baseline
```

Every report records the data range, the git SHA, the config hash and the random
seed, so a result can be regenerated exactly or shown to be irreproducible.
