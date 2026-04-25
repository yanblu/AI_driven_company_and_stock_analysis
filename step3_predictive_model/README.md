# Step 3 — Predictive Modelling

Predicts whether TD Bank will outperform, underperform, or be neutral relative to the XFN financials ETF over the next 5 trading days. Three separate XGBoost models (price, transcript, news) are combined via hard majority vote.

---

## Final Model

```
final_model/
├── artifacts/
│   ├── best_params_2026-04-23.json          ← Hyperparameters and feature lists
│   ├── xgb_ensemble_v2_price_2026-04-23.pkl
│   ├── xgb_ensemble_v2_transcript_2026-04-23.pkl
│   └── xgb_ensemble_v2_news_2026-04-23.pkl
├── notebooks/
│   ├── xgb_ensemble_v2_training.ipynb       ← Hyperparameter search, fold construction, artifact save
│   └── xgb_ensemble_v2_performance.ipynb   ← Walk-forward evaluation, SHAP analysis, confusion matrices
└── docs/
    ├── model_card.md          ← Model design, training setup, hyperparameters, performance tables
    ├── feature_engineering.md ← Feature definitions and formulas
    └── model_explainability.md ← SHAP observations, per-fold deep dive, when to trust the signal
```

### Performance (13 walk-forward folds, validated 2026-04-23)

| Model | Directional Call Rate | Directional Accuracy |
|---|---|---|
| Price | 0.972 | 0.530 |
| Transcript | 0.993 | 0.550 |
| News | 0.996 | 0.536 |
| **Hard Majority Vote** | **0.982** | **0.566** |

Random baseline = 0.50. Full per-fold breakdown and SHAP analysis in `notebooks/xgb_ensemble_v2_performance.ipynb`.

---

## To reproduce

The notebooks use `Path('').resolve()` to locate artifacts, so they must be executed with the working directory set to `final_model/notebooks/`. Run from the project root:

```bash
cd step3_predictive_model/final_model/notebooks

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 xgb_ensemble_v2_training.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 xgb_ensemble_v2_performance.ipynb

cd ../../..
```

The training notebook runs the hyperparameter grid search and saves `final_model/artifacts/best_params_*.json` and the three model pickles. The performance notebook loads those artifacts and must be run second.

---

## Archive

`model_experiments_archive/` — earlier experiment rounds (LightGBM, feature selection variants, V1 ensemble). Superseded by the final model above.
