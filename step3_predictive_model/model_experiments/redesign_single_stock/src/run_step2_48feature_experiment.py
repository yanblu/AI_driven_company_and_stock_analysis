"""
Step-2 grounded 48-feature static experiment.

Trains the same LightGBM multiclass setup as R07 / S2_Reduced (3-class,
`target_excess_xfn_5d`, +/-0.30% band, expanding-window E10 folds), but
uses the 48-feature static candidate pool defined in
`redesign_single_stock/nlp_event_feature_design.md` Section 9.

Compared to S2_Reduced (30 features), this pool:
  - adds 6 new Step-2 motivated event features (`evt_cfo_lead`,
    `evt_ceo_surprise`, `evt_analyst_qa_tone`, `evt_exec_scripted_gap`,
    `evt_source_dispersion`, `evt_transcript_reliability`)
  - drops `evt_news_tone` / `evt_news_flow` in favour of the raw rolling
    news features (no decay required, continuous through non-call periods)
  - drops `evt_macro_topic` (not in the Step-2 design)
  - expands the market/macro block to include price, technical, sector-
    relative, systematic-risk, global-risk, rates/macro, and news features

Outputs are written under `redesign_single_stock/data/step2_48/` and the
script prints a one-line side-by-side comparison against R07 and
S2_StaticShap30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_EXP_PARENT = Path(__file__).resolve().parents[2]  # model_experiments/
if str(_EXP_PARENT) not in sys.path:
    sys.path.insert(0, str(_EXP_PARENT))

from redesign_single_stock.src.run_redesign_experiments import (  # noqa: E402
    FEAT_PATH,
    PRICES_PATH,
    TARGET_THRESHOLD,
    baseline_majority,
    confusion_rows,
    decay_from_days,
    fit_predict,
    full_row_calibration,
    label_3class,
    load_price_returns,
    offset_metrics,
    Experiment,
)
from src.models.walk_forward_config import E10_FOLDS  # noqa: E402

OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_48"
EXP_ID = "STEP2_48F"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.0030

# ---------------------------------------------------------------------------
# Feature set definition — Section 9 of nlp_event_feature_design.md
# ---------------------------------------------------------------------------

MARKET_PRICE_TECH = [
    "td_return_1d",
    "td_return_5d",
    "td_return_20d",
    "td_volatility_20d",
    "td_volume_zscore_20d",
    "td_corr_pv_20d",
    "td_rsi_14",
    "td_bb_position",
    "td_dist_52w_high",
    "td_resi_20d",
]
MARKET_RELATIVE = [
    "td_vs_xfn_5d",
    "td_vs_tsx_5d",
    "xfn_return_5d",
]
SYSTEMATIC_RISK = [
    "td_beta_60d",
    "td_rsqr_60d",
    "td_resi_60d",
]
GLOBAL_RISK = [
    "vix_level",
    "vix_return_5d",
    "gold_return_5d",
    "dxy_return_5d",
    "fx_usdcad_return_5d",
]
RATES_MACRO = [
    "yield_10y_level",
    "yield_curve_slope",
    "yield_curve_slope_change_5d",
    "rate_overnight_level",
    "cpi_yoy_change",
]
NEWS_FLOW = [
    "news_sent_mean_30d",
    "news_count_30d",
    "is_news_burst",
]
TIMING = [
    "days_since_call",
    "is_earnings_week",
    "days_since_last_news",
]

# 16 Step-2 motivated NLP event features
NLP_EVENT = [
    "evt_ceo_tone",
    "evt_cfo_tone",
    "evt_cfo_lead",              # new
    "evt_ceo_surprise",          # new
    "evt_analyst_qa_tone",       # new
    "evt_exec_scripted_gap",     # new
    "evt_framing_gap",
    "evt_guidance_strength",
    "evt_guidance_shift",
    "evt_aml_pressure",
    "evt_aml_shift",
    "evt_source_dispersion",     # new
    "evt_topic_entropy",
    "evt_broad_topic_signal",
    "evt_broad_topic_shift",
    "evt_transcript_reliability",  # new
]

FEATURES_48 = (
    MARKET_PRICE_TECH
    + MARKET_RELATIVE
    + SYSTEMATIC_RISK
    + GLOBAL_RISK
    + RATES_MACRO
    + NEWS_FLOW
    + TIMING
    + NLP_EVENT
)

FEATURE_GROUPS: Dict[str, List[str]] = {
    "A_td_price_tech": MARKET_PRICE_TECH,
    "B_sector_market_relative": MARKET_RELATIVE,
    "C_systematic_risk": SYSTEMATIC_RISK,
    "D_global_risk": GLOBAL_RISK,
    "E_rates_macro": RATES_MACRO,
    "F_news_flow": NEWS_FLOW,
    "G_timing": TIMING,
    "H_nlp_event": NLP_EVENT,
}

NEW_EVT_FEATURES = [
    "evt_cfo_lead",
    "evt_ceo_surprise",
    "evt_analyst_qa_tone",
    "evt_exec_scripted_gap",
    "evt_source_dispersion",
    "evt_transcript_reliability",
]


# ---------------------------------------------------------------------------
# Dataset loader that adds the 6 new Step-2 event features
# ---------------------------------------------------------------------------


def load_dataset_step2() -> pd.DataFrame:
    """Daily matrix augmented with the 6 new Step-2 NLP event features.

    Re-implements `run_redesign_experiments.load_dataset` but adds the six
    new `evt_*` primitives motivated in `nlp_event_feature_design.md`.
    """
    df = pd.read_parquet(FEAT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat_cols = [c for c in df.columns if c != "date"]
    df[feat_cols] = df[feat_cols].replace(-9999.0, np.nan)

    td = load_price_returns("TD_TO").rename(columns={"td_to_fwd5": "td_fwd5"})
    xfn = load_price_returns("XFN_TO").rename(columns={"xfn_to_fwd5": "xfn_fwd5"})
    tsx = load_price_returns("GSPTSE").rename(columns={"gsptse_fwd5": "tsx_fwd5"})
    df = df.merge(td, on="date", how="left")
    df = df.merge(xfn, on="date", how="left")
    df = df.merge(tsx, on="date", how="left")

    engineered: Dict[str, pd.Series] = {}
    engineered["target_excess_xfn_5d"] = df["td_fwd5"] - df["xfn_fwd5"]
    engineered["target_excess_tsx_5d"] = df["td_fwd5"] - df["tsx_fwd5"]

    call_decay = decay_from_days(df["days_since_call"])

    ceo = df["transcript_ceo_prep_sentiment_mean_ffill"]
    cfo = df["transcript_cfo_prep_sentiment_mean_ffill"]
    exec_qa = df["transcript_exec_qa_sentiment_mean_ffill"]
    analyst_qa = df["transcript_analyst_qa_sentiment_mean_ffill"]
    sent_std = df["transcript_sentiment_std_ffill"]

    engineered["evt_ceo_tone"] = ceo * call_decay
    engineered["evt_cfo_tone"] = cfo * call_decay

    engineered["evt_cfo_lead"] = (cfo - ceo) * call_decay
    engineered["evt_ceo_surprise"] = (
        df["transcript_ceo_prep_sentiment_mean_surprise"] * call_decay
    )
    engineered["evt_analyst_qa_tone"] = analyst_qa * call_decay
    engineered["evt_exec_scripted_gap"] = (exec_qa - ceo) * call_decay

    engineered["evt_framing_gap"] = df["framing_gap_ffill"] * call_decay

    engineered["evt_aml_pressure"] = (
        df["topic_regulatory_AML_share_ffill"]
        * (1.0 - df["topic_regulatory_AML_sentiment_ffill"])
    ) * call_decay
    engineered["evt_aml_shift"] = (
        df["topic_regulatory_AML_share_delta"].abs()
        + df["topic_regulatory_AML_sentiment_delta"].abs()
    ) * call_decay

    # Cross-source consistency: std of CEO / news / filing sentiment.
    # Low std ⇒ channels agree (regime shift); high std ⇒ single-source noise.
    stacked_sources = pd.concat(
        [
            ceo,
            df["news_sentiment_mean_ffill"],
            df["report_sentiment_mean_ffill"],
        ],
        axis=1,
    )
    engineered["evt_source_dispersion"] = stacked_sources.std(axis=1, skipna=True) * call_decay

    engineered["evt_guidance_strength"] = (
        df["topic_guidance_share_ffill"] * df["topic_guidance_sentiment_ffill"]
    ) * call_decay
    engineered["evt_guidance_shift"] = (
        df["topic_guidance_share_delta"] + df["topic_guidance_sentiment_delta"]
    ) * call_decay

    engineered["evt_topic_entropy"] = df["topic_entropy_ffill"] * call_decay

    broad_topic_signal = [
        df["topic_credit_quality_share_ffill"] * (df["topic_credit_quality_sentiment_ffill"] - 0.5),
        df["topic_capital_share_ffill"] * (df["topic_capital_sentiment_ffill"] - 0.5),
        df["topic_wealth_wholesale_share_ffill"] * (df["topic_wealth_wholesale_sentiment_ffill"] - 0.5),
        df["topic_US_retail_share_ffill"] * (df["topic_US_retail_sentiment_ffill"] - 0.5),
        df["topic_NIM_share_ffill"] * (df["topic_NIM_sentiment_ffill"] - 0.5),
    ]
    engineered["evt_broad_topic_signal"] = pd.concat(broad_topic_signal, axis=1).mean(axis=1) * call_decay

    broad_topic_shift = [
        df["topic_credit_quality_share_delta"].abs() + df["topic_credit_quality_sentiment_delta"].abs(),
        df["topic_capital_share_delta"].abs() + df["topic_capital_sentiment_delta"].abs(),
        df["topic_wealth_wholesale_share_delta"].abs() + df["topic_wealth_wholesale_sentiment_delta"].abs(),
        df["topic_US_retail_share_delta"].abs() + df["topic_US_retail_sentiment_delta"].abs(),
        df["topic_NIM_share_delta"].abs() + df["topic_NIM_sentiment_delta"].abs(),
    ]
    engineered["evt_broad_topic_shift"] = pd.concat(broad_topic_shift, axis=1).mean(axis=1) * call_decay

    engineered["evt_transcript_reliability"] = (1.0 / (1.0 + sent_std)) * call_decay

    engineered_df = pd.DataFrame(engineered, index=df.index)
    return pd.concat([df, engineered_df], axis=1).copy()


# ---------------------------------------------------------------------------
# Walk-forward run
# ---------------------------------------------------------------------------


def run() -> Dict[str, object]:
    df = load_dataset_step2()

    missing = [c for c in FEATURES_48 if c not in df.columns]
    if missing:
        raise RuntimeError(f"Features missing from dataset: {missing}")

    exp = Experiment(
        exp_id=EXP_ID,
        stage="step2_48feature",
        title="Step-2 grounded 48-feature static LightGBM multiclass",
        target=TARGET,
        feature_mode="manual_static",
        model_kind="lgbm_multiclass",
        threshold=THRESHOLD,
    )

    fold_rows: List[Dict[str, object]] = []
    confusion_all: List[Dict[str, object]] = []

    for fold_id, train_end, test_start, test_end in E10_FOLDS:
        train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[TARGET].notna()
        test_mask = (
            (df["date"] >= pd.Timestamp(test_start))
            & (df["date"] <= pd.Timestamp(test_end))
            & df[TARGET].notna()
        )
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if len(train_df) < 80 or len(test_df) < 20:
            continue

        x_train = train_df[FEATURES_48].copy()
        x_test = test_df[FEATURES_48].copy()
        y_train_cont = train_df[TARGET].values
        y_test_cont = test_df[TARGET].values
        y_train_cls = label_3class(y_train_cont, threshold=THRESHOLD)
        y_test_cls = label_3class(y_test_cont, threshold=THRESHOLD)

        y_pred_cls = fit_predict(exp, x_train, y_train_cls, x_test)

        metrics = offset_metrics(y_test_cont, y_test_cls, y_pred_cls)
        maj_pred = baseline_majority(y_train_cls, len(y_test_cls))
        mom_pred = label_3class(test_df["td_vs_xfn_5d"].values, threshold=THRESHOLD)
        maj_metrics = offset_metrics(y_test_cont, y_test_cls, maj_pred)
        mom_metrics = offset_metrics(y_test_cont, y_test_cls, mom_pred)
        calibration = full_row_calibration(y_test_cls, y_pred_cls)
        confusion_all.extend(confusion_rows(EXP_ID, fold_id, y_test_cls, y_pred_cls))

        fold_rows.append(
            {
                "exp_id": EXP_ID,
                "fold": fold_id,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_count": len(FEATURES_48),
                "threshold": THRESHOLD,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "active_sign_acc": metrics["active_sign_acc"],
                "active_coverage": metrics["active_coverage"],
                "majority_accuracy": maj_metrics["accuracy"],
                "momentum_accuracy": mom_metrics["accuracy"],
                **calibration,
            }
        )

    folds_df = pd.DataFrame(fold_rows)

    summary = {
        "exp_id": EXP_ID,
        "feature_count": len(FEATURES_48),
        "threshold": THRESHOLD,
        "n_folds": int(len(folds_df)),
        "mean_accuracy": float(folds_df["accuracy"].mean()),
        "mean_macro_f1": float(folds_df["macro_f1"].mean()),
        "mean_active_sign_acc": float(folds_df["active_sign_acc"].mean()),
        "mean_active_coverage": float(folds_df["active_coverage"].mean()),
        "mean_majority_accuracy": float(folds_df["majority_accuracy"].mean()),
        "mean_momentum_accuracy": float(folds_df["momentum_accuracy"].mean()),
        "mean_active_precision_pos": float(folds_df["active_precision_pos"].mean()),
        "mean_active_precision_neg": float(folds_df["active_precision_neg"].mean()),
        "mean_predicted_max_share": float(folds_df["predicted_max_share"].mean()),
    }
    summary["uplift_vs_majority_pp"] = 100 * (
        summary["mean_accuracy"] - summary["mean_majority_accuracy"]
    )
    summary["uplift_vs_momentum_pp"] = 100 * (
        summary["mean_accuracy"] - summary["mean_momentum_accuracy"]
    )

    feature_manifest = {
        "exp_id": EXP_ID,
        "target": TARGET,
        "threshold": THRESHOLD,
        "feature_count": len(FEATURES_48),
        "feature_groups": FEATURE_GROUPS,
        "new_event_features": NEW_EVT_FEATURES,
        "features": FEATURES_48,
        "notes": (
            "48-feature static pool defined in Section 9 of "
            "redesign_single_stock/nlp_event_feature_design.md. Trained with "
            "same LightGBM hyperparameters and walk-forward as R07 / S2_Reduced."
        ),
    }

    return {
        "folds_df": folds_df,
        "summary": summary,
        "feature_manifest": feature_manifest,
        "confusion_df": pd.DataFrame(confusion_all),
    }


def _load_baseline_row(exp_id: str) -> pd.Series:
    df = pd.read_csv(ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/round2_summary.csv")
    row = df.loc[df["exp_id"] == exp_id]
    if row.empty:
        raise RuntimeError(f"{exp_id} not found in round2_summary.csv")
    return row.iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = run()

    folds_path = OUT_DIR / "step2_48_fold_metrics.csv"
    summary_path = OUT_DIR / "step2_48_summary.json"
    manifest_path = OUT_DIR / "step2_48_feature_manifest.json"
    confusion_path = OUT_DIR / "step2_48_confusion.csv"

    out["folds_df"].to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(out["summary"], indent=2))
    manifest_path.write_text(json.dumps(out["feature_manifest"], indent=2))
    out["confusion_df"].to_csv(confusion_path, index=False)

    r07 = _load_baseline_row("S2_Reduced")
    shap30 = _load_baseline_row("S2_StaticShap30")
    summary = out["summary"]

    print("\n=== Step-2 48-feature static model ===")
    print(f"feature_count        : {summary['feature_count']}")
    print(f"threshold            : {summary['threshold']:.4f} (+/- 0.30%)")
    print(f"folds evaluated      : {summary['n_folds']}")
    print()
    print(f"{'metric':<28} {'STEP2_48F':>11} {'R07/S2_Reduced':>16} {'S2_StaticShap30':>17}")
    print("-" * 77)

    def row(name: str, key: str, r07_key: str | None = None) -> None:
        r07_k = r07_key or key
        print(
            f"{name:<28} {summary[key]:>11.4f} "
            f"{float(r07[r07_k]):>16.4f} {float(shap30[r07_k]):>17.4f}"
        )

    row("mean_accuracy", "mean_accuracy")
    row("mean_macro_f1", "mean_macro_f1")
    row("mean_active_sign_acc", "mean_active_sign_acc")
    row("mean_active_coverage", "mean_active_coverage")
    row("mean_active_precision_pos", "mean_active_precision_pos")
    row("mean_active_precision_neg", "mean_active_precision_neg")
    row("mean_majority_accuracy", "mean_majority_accuracy")
    row("mean_momentum_accuracy", "mean_momentum_accuracy")
    print()
    print(f"uplift vs majority   : {summary['uplift_vs_majority_pp']:+.2f} pp")
    print(f"uplift vs momentum   : {summary['uplift_vs_momentum_pp']:+.2f} pp")
    print()
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {summary_path}")
    print(f"Saved -> {manifest_path}")
    print(f"Saved -> {confusion_path}")


if __name__ == "__main__":
    main()
