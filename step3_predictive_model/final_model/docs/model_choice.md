# Model Choice

Key design decisions made during model development.

---

## Target: `target_excess_xfn_5d`

**Why sector-relative?**
TD's raw return is driven largely by sector-wide forces (rates, credit cycle, macro) that move all Canadian banks together. Predicting raw price conflates sector beta with TD-specific alpha. Subtracting XFN's return isolates the TD-specific component — earnings quality, management tone, AML exposure — which is the actual signal of interest.

**Why XFN?**
XFN (iShares S&P/TSX Capped Financials ETF) is TD's most direct liquid peer benchmark. Its top holdings are the Big Five Canadian banks, with TD typically the second-largest constituent (~18–20%) after RBC (~25%), followed by BNS, BMO, and CM. Using XFN means the excess return measures how TD performs relative to its own peer group, not a broad market index.

| Approximate top holdings | Weight |
|---|---|
| Royal Bank (RY) | ~25% |
| TD Bank (TD) | ~19% |
| Scotiabank (BNS) | ~11% |
| Bank of Montreal (BMO) | ~10% |
| CIBC (CM) | ~8% |
| Manulife (MFC) | ~6% |
| Sun Life (SLF) | ~4% |
| Other financials | ~17% |

*Weights are approximate and rebalance periodically. Source: iShares Canada.*

**Why 5-day horizon?**
Short enough to be actionable; long enough for sentiment and momentum signals to register. 1–2 day horizons are dominated by microstructure noise; 20+ day horizons span multiple retraining cycles and mix signals from different quarters.

**Why ±0.3% threshold?**
At ±0.3%, roughly 30–35% of days are neutral — a meaningful but not dominant class. Too tight inflates the active class and degrades precision; too wide pushes genuine directional moves into neutral.

---

## Algorithm: XGBoost

~800 training rows per fold, 5–10 features per model. Overfitting is the primary risk.

- **vs. LightGBM**: tested on this dataset, no meaningful performance advantage.
- **vs. logistic regression**: non-linear interactions between sentiment, momentum, and the target require a non-linear model.
- **vs. neural networks**: inappropriate for this data scale.

---

## Architecture: Three-Model Ensemble

Each sub-model is trained on one information source only — price, transcript, or news. This design has three advantages:

1. **Update frequency separation**: price features update daily; transcript features update quarterly; news features are event-driven. A single model would need to handle mixed-frequency signals simultaneously, creating noisy gradient updates.
2. **Interpretability**: each model's predictions and SHAP attributions can be inspected independently. If the ensemble degrades, the failing signal source can be identified and fixed in isolation.
3. **Scalability**: adding a new information source (e.g. options flow, analyst revisions) means adding a new sub-model without modifying the existing three. The ensemble logic remains unchanged.

---

## Ensemble Method: Hard Majority Vote

Hard majority vote achieved 0.566 directional accuracy vs 0.536 for soft-prob averaging. The gap indicates the models know *direction* reliably but not *confidence* — probability estimates are not well-calibrated. Hard voting discards the miscalibrated confidence and uses only the directional component.

---

## Training Window and Retraining Cadence

- **2-year rolling window (504 days)**: captures at least 8 quarterly earnings cycles — necessary for the transcript model to learn call-to-call patterns. Shorter windows risk fitting a single regime.
- **Quarterly retraining**: aligns with the earnings cycle; ensures the most recent call is in the training set.
- **5-day gap**: enforced between train end and test start to prevent any overlap between price-based features and the forward return target.

---

## Hyperparameter Grid

Designed to control overfitting on a small dataset: shallow trees (`max_depth` ≤ 4), high leaf size (`min_child_weight` 10–30), and explicit L1/L2 regularisation. The news model selected the strongest regularisation (`max_depth=2`, `reg_lambda=2.0`), consistent with its 5-feature complexity.

Grid evaluated on folds 4–10 to avoid edge folds (minimum training size at the start; most recent quarter at the end).

---

## What Was Tested and Rejected

| Feature / Choice | Reason Rejected |
|---|---|
| 60-day price returns | Quarterly retraining captures regime shifts; 60d adds noise at the 5-day horizon |
| Volume change features | No directional mechanism; empirically reduced performance |
| 90-day news window | Dilutes short-horizon signal without a clear mechanism |
| `call_decay` on NLP features | Caused near-perfect multicollinearity; replaced by `days_since_call` |
| Macro features (FX, CPI, rates) | Already captured in XFN benchmark; double-counts sector signal |
| Single combined model | Loss of interpretability; mixed update frequencies |
| Soft-prob averaging | Underperformed hard vote; models are not probability-calibrated |
