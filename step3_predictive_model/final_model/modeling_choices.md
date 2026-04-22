# lgbm_3class_xfn5d_30f — Modeling Choices

This document explains why the final model is configured the way it is, tracing the key decisions through the experiment history that produced it. The model was labelled **R07** in the experiment log. The full experiment log lives in `model_experiments/redesign_single_stock/redesign_single_stock_experiments.md`.

---

## Round 1 — systematic experiment across all four dimensions

Seven configurations (R01–R07) were tested in Round 1, crossing four design dimensions: target definition, neutral band width, feature set size, and model family.

### Acceptance rule

A candidate was only marked acceptable if **all** of the following held:
- Mean accuracy lift vs majority baseline > 2pp
- Mean accuracy lift vs momentum baseline > 1pp
- Macro F1 > 0.34
- Active sign accuracy > 50%

The macro F1 threshold catches models that appear accurate by being very biased toward one class. The active sign accuracy threshold ensures the model has genuine directional skill on its non-neutral predictions.

### Configurations tested

| ID | Target | Neutral band | Features | Model |
|---|---|---|---|---|
| R01 | Excess vs XFN | ±0.5% | Reduced 30 | Logistic regression |
| R02 | Excess vs XFN | ±0.5% | Reduced 30 | Elastic net → thresholded |
| R03 | Excess vs XFN | ±0.5% | Reduced 30 | Shallow LightGBM |
| R04 | Excess vs XFN | ±0.5% | Full 180 | Shallow LightGBM |
| R05 | Excess vs TSX | ±0.5% | Reduced 30 | Logistic regression |
| R06 | Excess vs XFN | **±0.3%** | Reduced 30 | Logistic regression |
| **R07** | **Excess vs XFN** | **±0.3%** | **Reduced 30** | **Shallow LightGBM** |

### Round 1 results

R07 performance is from the verified 7-fold walk-forward re-run in `lgbm_3class_xfn5d_30f_performance_and_shap.ipynb`. All other rows are from the original Round 1 experiment run.

| ID | Target | Band | Features | Model | Mean acc | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **R07** | Excess vs XFN | **±0.3%** | **Reduced 30** | **LightGBM** | **46.8%** | **0.359** | **60.0%** | **94.3%** | **Winner ✓** |
| R06 | Excess vs XFN | ±0.3% | Reduced 30 | Logistic reg | 38.4% | 0.302 | 53.8% | 81.7% | No |
| R04 | Excess vs XFN | ±0.5% | Full 180 | LightGBM | 33.9% | 0.317 | 50.3% | 71.4% | No |
| R03 | Excess vs XFN | ±0.5% | Reduced 30 | LightGBM | 35.8% | 0.313 | 53.5% | 74.7% | No |
| R05 | Excess vs TSX | ±0.5% | Reduced 30 | Logistic reg | 36.2% | 0.290 | 50.4% | 88.6% | No |
| R01 | Excess vs XFN | ±0.5% | Reduced 30 | Logistic reg | 36.2% | 0.282 | 52.6% | 91.8% | No |
| R02 | Excess vs XFN | ±0.5% | Reduced 30 | Elastic net | 35.0% | 0.170 | — | 0.0% | No |

**Acceptance criteria** (all four must pass): mean accuracy lift vs majority BL > 2pp · mean accuracy lift vs momentum BL > 1pp · macro F1 > 0.34 · active sign accuracy > 50%. Only R07 passed all four. The key decisions it validated:

---

## Decision 1 — Target: excess return vs XFN, not raw return

Comparing R01 (excess vs XFN) against R05 (excess vs TSX):
- Both use the same model (logistic regression) and the same 30 features
- R01 outperforms R05 on every metric
- XFN is the financials sector ETF — it is much closer to TD's systematic return driver than the broad TSX index
- Removing sector-common variance makes the residual (what TD does differently) more predictable

**Conclusion**: `target_excess_xfn_5d` is the better-defined target.

---

## Decision 2 — Neutral band: ±0.3%, not ±0.5%

Comparing R06 and R07 (tight band) against R01 and R03 (standard band):
- Tightening the neutral band from ±0.5% to ±0.3% consistently improves active sign accuracy and macro F1
- At ±0.5%, the neutral class absorbs too many days that are directionally informative; the model learns to hedge into neutral
- At ±0.3%, the neutral class is genuinely ambiguous, while outperform/underperform labels are more crisply defined — the model has a cleaner signal to learn

**Conclusion**: the ±0.3% neutral band produces better-calibrated, more actionable signals.

---

## Decision 3 — Feature set: reduced 30, not full 180

Comparing R04 (full 180 features, LightGBM) against R03 (reduced 30, same model and band):
- R04 drops to 33.9% mean accuracy vs 35.8% for R03
- R04 active sign accuracy is only 50.3% — barely above chance
- Coverage is also lower at 71.4%, meaning the model abstains more but is still wrong when it does act

With ~1,285 daily training rows and 180 features, the full-feature model overfits: it finds spurious patterns in the 99-column NLP block (mostly forward-filled quarterly values) that don't generalise. The reduced 30-feature set has a much healthier row-to-feature ratio and focuses on the most economically motivated signals.

**Conclusion**: fewer, better-motivated features beat a wide feature space for a single-stock setting.

---

## Decision 4 — Model family: LightGBM, not logistic regression

Comparing R07 (LightGBM) against R06 (logistic regression, identical target and features):
- R07 mean accuracy: 46.8% vs R06: 38.4%
- R07 macro F1: 0.359 vs R06: 0.302
- R07 active sign accuracy: 60.0% vs R06: 53.8%
- R07 coverage: 94.3% vs R06: 81.7%

Logistic regression's linear decision boundary is too restrictive for the non-linear interactions between market regime, NLP event signals, and timing — for example, AML narrative pressure matters differently in a rising vs falling rate environment. LightGBM's shallow tree structure captures these interactions without requiring hand-engineered interaction terms.

The model is kept deliberately **shallow** (`num_leaves=8`, `n_estimators=120`) to limit overfitting. Standard L1/L2 regularisation (`reg_alpha=0.2`, `reg_lambda=1.0`) is also applied.

**Conclusion**: shallow LightGBM is the right model family for this problem.

---

## XGBoost comparison (from Round 2)

After selecting R07 as the best configuration, XGBoost was also tested as a direct substitute for LightGBM, using the same 30 features and tighter bands:

| ID | Model | Neutral band | Mean acc | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| **R07 / lgbm_3class_xfn5d_30f** | **LightGBM** | **±0.30%** | **46.8%** | **0.359** | **60.0%** | **94.3%** | **Winner** |
| S3_XGB_MULTICLASS | XGBoost | ±0.30% | 42.7% | 0.309 | 53.8% | 97.0% | No |
| S1_XGB_B25 | XGBoost | ±0.25% | 44.5% | 0.304 | 54.0% | 99.3% | No |
| S1_XGB_B20 | XGBoost | ±0.20% | 44.7% | 0.297 | 52.8% | 100.0% | No |

Key observations:
- **XGBoost at ±0.30%** (`S3_XGB_MULTICLASS`) has higher coverage but lower accuracy, macro F1, and active sign accuracy than LightGBM — it is more active but less precise
- **XGBoost at tighter bands (±0.25% and ±0.20%)** both improve raw accuracy slightly relative to the wider XGBoost band, but at the cost of class balance (macro F1 below 0.31) and active sign accuracy
- At ±0.20%, coverage reaches 100% — the model essentially stops predicting neutral, which inflates activity counts without improving direction quality
- LightGBM's gradient-boosting implementation appears to better regularise the multiclass boundary on this small dataset (fewer than 1,300 daily rows)

**Conclusion**: LightGBM outperforms XGBoost on the metrics that matter for pension signal quality (macro F1 and active sign accuracy). Swapping the model family does not improve outcomes; the task design and feature set matter more than the tree boosting implementation.

---

## Final hyperparameter choices

| Parameter | Value | Reason |
|---|---|---|
| `num_leaves` | 8 | Shallow trees — avoids memorising single-stock regime patterns |
| `n_estimators` | 120 | Enough boosting rounds to fit signal; early stopping not needed given `min_child_samples` constraint |
| `learning_rate` | 0.05 | Conservative learning rate; works well with shallow trees |
| `min_child_samples` | 40 | Prevents any leaf from fitting on fewer than 40 rows (~3% of training data); strong overfitting guard |
| `feature_fraction` | 0.8 | Subsample features per tree for variance reduction |
| `reg_alpha` | 0.2 | L1 sparsity regularisation |
| `reg_lambda` | 1.0 | L2 weight-decay regularisation |
| `random_state` | 42 | Fixed seed for reproducibility |

---

## Summary: why lgbm_3class_xfn5d_30f is the final model

This configuration (R07 in the experiment log) is the only one that simultaneously solves all four design problems:

1. **Target** — relative to sector (XFN), removing positive-drift bias
2. **Decision framing** — tight ±0.3% neutral band giving clean labels
3. **Features** — 30 event-style features with the right row-to-feature ratio for single-stock data
4. **Model** — shallow LightGBM that outperforms all alternatives tested (logistic regression, elastic net, XGBoost)

No subsequent experiment across Rounds 2–6 produced a configuration that beat R07 on the business-relevant metrics (active sign accuracy, macro F1) while remaining methodologically sound. R07 is therefore selected as the final model and saved as `lgbm_3class_xfn5d_30f`.
