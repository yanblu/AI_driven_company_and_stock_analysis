# Step 3 — Predictive Modelling

[← Project overview](../README.md) · [← Step 2: LLM analysis](../step2_llm_analysis/README.md) · [Case-study deck](../docs/presentation/summary-slide.pdf)

Tests whether market and language signals can help predict if TD Bank will outperform, underperform, or remain approximately neutral relative to the XFN financials ETF over the next five trading days.

## At a glance

| | Description |
|---|---|
| **Purpose** | Combine structured market data and LLM-derived features in a time-aware predictive experiment |
| **Inputs** | Daily price, transcript, filing-topic, and rolling news features produced by Steps 1 and 2 |
| **Model** | Three XGBoost classifiers—price, transcript, and news—combined through hard majority vote |
| **Target** | Direction of TD's five-day excess return relative to XFN, with a neutral band of ±0.3% |
| **Evaluation** | Quarterly walk-forward folds with a rolling two-year training window and a five-trading-day gap |
| **Outputs** | Saved classifiers, feature definitions, fold-level results, and SHAP explanations |

For the quickest technical review, start with the [model card](./final_model/docs/model_card.md). The [performance notebook](./final_model/notebooks/xgb_ensemble_v2_performance.ipynb) contains the complete walk-forward analysis, and [model explainability](./final_model/docs/model_explainability.md) summarizes the SHAP findings.

## Selected model

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

### Reported performance across 13 walk-forward folds

| Model | Directional call rate | Directional accuracy |
|---|---|---|
| Price | 97.2% | 53.0% |
| Transcript | 99.3% | 55.0% |
| News | 99.6% | 53.6% |
| **Hard majority vote** | **98.2%** | **56.6%** |

Directional accuracy measures whether a non-neutral prediction correctly identifies the sign of the five-day excess return. These results are exploratory evidence from a single-company case study; see the [model card](./final_model/docs/model_card.md) for baseline comparisons, fold-level results, and limitations.

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

## Archive

Earlier LightGBM, feature-selection, and ensemble experiments are retained in the [model experiments archive](./model_experiments_archive/README.md) for traceability. They are superseded by the selected model above and are not the recommended starting point.
