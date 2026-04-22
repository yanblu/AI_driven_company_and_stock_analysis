# Step 3 — Predictive Modeling

Stock signal detection model for TD Bank using LightGBM multiclass classification.

## Structure

```
step3_predictive_model/
├── final_model/                    ← Production-ready model
│   ├── artifacts/
│   │   ├── lgbm_3class_xfn5d_30f_tuned-*.pkl  ← Tuned model artifact (final)
│   │   └── lgbm_3class_xfn5d_30f_tuned-*.json ← Feature list, params, metadata
│   ├── model_card.md               ← Full model documentation
│   ├── lgbm_3class_xfn5d_30f_training.ipynb          ← Hyperparameter search + artifact save
│   ├── lgbm_3class_xfn5d_30f_performance_and_shap.ipynb  ← Walk-forward evaluation + SHAP
│   ├── feature_engineering.md      ← How each of the 30 features is constructed
│   └── modeling_choices.md         ← Experiment decisions leading to final config
│
└── model_experiments/              ← Full experiment history
    ├── redesign_single_stock/      ← All Python experiment code + results
    │   ├── src/                    ← Experiment runner scripts
    │   ├── data/                   ← Experiment outputs (CSVs, JSONs, artifacts)
    │   ├── redesign_single_stock_experiments.md  ← Experiment log (6 rounds)
    │   └── nlp_event_feature_design.md           ← NLP feature design doc
    ├── notebooks/                  ← Original modeling notebooks
    │   ├── 03_modeling.ipynb
    │   ├── 03a_data_quality.ipynb
    │   └── 04_final_model.ipynb    ← Original (pre-redesign) model
    └── docs/                       ← Step 3 planning docs and SHAP visuals
```

## Final Model: lgbm_3class_xfn5d_30f

**Quick summary:**
- Predicts whether TD Bank will **outperform**, **underperform**, or be **neutral** relative to the XFN financials sector ETF over the next 5 trading days
- Uses 30 features: 19 market/macro/timing + 11 LLM-derived event features from earnings calls
- Produces a **directional signal** (60% active-sign accuracy, +21pp over baselines)
- See `final_model/model_card.md` for full documentation

## Experiment History (6 Rounds)

| Round | Focus | Best Result |
|---|---|---|
| R1 | Initial redesign: new target (excess return), 3-class framing, R07 baseline | R07: 60% active sign acc |
| R2 | Band sweep, XGBoost/LightGBM comparison, static SHAP feature selection | S2_StaticShap30: 56.8% (leakage caveat) |
| R3 | Leakage-clean feature selection (fold-local SHAP, bootstrap SHAP) | SHAP bootstrap: 54% — clean but lower |
| R4 | Step-2 grounded 48-feature static pool | STEP2_48F: strong on AML fold, weaker overall |
| R5 | Bootstrap SHAP on 48-feature Step-2 pool | STEP2_48F_Boot30: best leakage-clean candidate |
| R6 | Global static SHAP on full pool + Step-2 features | STEP2_Full_StaticShap30: no improvement |

**Verdict**: R07 (saved as `lgbm_3class_xfn5d_30f`) selected as final model. Best empirical performance across all 7 folds. Feature selection is domain-justified (see experiment log Round 1 and NLP feature design doc).
