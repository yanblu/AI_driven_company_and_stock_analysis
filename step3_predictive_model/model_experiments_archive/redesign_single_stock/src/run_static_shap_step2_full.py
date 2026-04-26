"""
Global static SHAP top-30 on the full pool + Step-2 inspired event features.

Same methodology as `S2_StaticShap30` in the round-2 pipeline:

- candidate pool = every non-target column in the daily parquet, including
  all raw NLP `_ffill` / `_delta` / `_surprise` columns, all market / macro
  columns, all original `evt_*` features, plus the six new Step-2 inspired
  primitives (`evt_cfo_lead`, `evt_ceo_surprise`, `evt_analyst_qa_tone`,
  `evt_exec_scripted_gap`, `evt_source_dispersion`,
  `evt_transcript_reliability`)
- global static SHAP ranking aggregated across all outer training folds
- take the top-30 features, apply the same static list to every fold,
  retrain LightGBM multiclass and evaluate on each fold's test window

This reproduces the same feature-selection look-ahead flavour as
`S2_StaticShap30` (the feature list is chosen based on SHAP aggregated
across every fold's training data), so the numbers are directly
comparable to the current locked redesign model. Not leakage-clean.

Outputs: `redesign_single_stock/data/step2_full_staticshap/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_EXP_PARENT = Path(__file__).resolve().parents[2]  # model_experiments/
if str(_EXP_PARENT) not in sys.path:
    sys.path.insert(0, str(_EXP_PARENT))

from redesign_single_stock.src.run_redesign_experiments import (  # noqa: E402
    Experiment,
    baseline_majority,
    confusion_rows,
    decay_from_days,
    derive_stable_core_features,
    feature_set_full,
    fit_predict,
    full_row_calibration,
    label_3class,
    load_dataset,
    offset_metrics,
)
from src.models.walk_forward_config import E10_FOLDS  # noqa: E402


OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_full_staticshap"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.0030
TOP_K = 30
EXP_ID = "STEP2_Full_StaticShap30"
TITLE = "Global static SHAP top-30 over full pool + 6 new Step-2 event features"


def add_step2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the six new Step-2 inspired event features to the base dataset."""
    out = df.copy()
    call_decay = decay_from_days(out["days_since_call"])

    ceo = out["transcript_ceo_prep_sentiment_mean_ffill"]
    cfo = out["transcript_cfo_prep_sentiment_mean_ffill"]
    exec_qa = out["transcript_exec_qa_sentiment_mean_ffill"]
    analyst_qa = out["transcript_analyst_qa_sentiment_mean_ffill"]
    sent_std = out["transcript_sentiment_std_ffill"]

    extras: Dict[str, pd.Series] = {}
    extras["evt_cfo_lead"] = (cfo - ceo) * call_decay
    extras["evt_ceo_surprise"] = (
        out["transcript_ceo_prep_sentiment_mean_surprise"] * call_decay
    )
    extras["evt_analyst_qa_tone"] = analyst_qa * call_decay
    extras["evt_exec_scripted_gap"] = (exec_qa - ceo) * call_decay

    stacked_sources = pd.concat(
        [
            ceo,
            out["news_sentiment_mean_ffill"],
            out["report_sentiment_mean_ffill"],
        ],
        axis=1,
    )
    extras["evt_source_dispersion"] = (
        stacked_sources.std(axis=1, skipna=True) * call_decay
    )
    extras["evt_transcript_reliability"] = (1.0 / (1.0 + sent_std)) * call_decay

    extras_df = pd.DataFrame(extras, index=out.index)
    return pd.concat([out, extras_df], axis=1)


NEW_EVT_FEATURES = [
    "evt_cfo_lead",
    "evt_ceo_surprise",
    "evt_analyst_qa_tone",
    "evt_exec_scripted_gap",
    "evt_source_dispersion",
    "evt_transcript_reliability",
]

EXISTING_EVT_FEATURES = [
    "evt_ceo_tone",
    "evt_cfo_tone",
    "evt_framing_gap",
    "evt_aml_pressure",
    "evt_aml_shift",
    "evt_guidance_strength",
    "evt_guidance_shift",
    "evt_macro_topic",
    "evt_topic_entropy",
    "evt_news_tone",
    "evt_news_flow",
    "evt_broad_topic_signal",
    "evt_broad_topic_shift",
]


def run_experiment() -> Dict[str, object]:
    base_df = load_dataset()
    df = add_step2_features(base_df)

    candidate_pool = feature_set_full(df)
    print(
        f"Candidate pool size = {len(candidate_pool)} "
        f"(full daily parquet + {len(EXISTING_EVT_FEATURES)} existing evt_* + "
        f"{len(NEW_EVT_FEATURES)} new Step-2 evt_*)."
    )

    # Global static SHAP ranking — same as S2_StaticShap30.
    # derive_stable_core_features runs SHAP on every fold's training window
    # and aggregates. Setting core_size = TOP_K returns a top-TOP_K list.
    top_features, stats_df = derive_stable_core_features(
        df=df,
        target=TARGET,
        threshold=THRESHOLD,
        core_size=TOP_K,
    )
    print(f"Derived global static SHAP top-{TOP_K} list.")

    exp = Experiment(
        exp_id=EXP_ID,
        stage="step2_full_staticshap",
        title=TITLE,
        target=TARGET,
        feature_mode="static_shap_full_pool",
        model_kind="lgbm_multiclass",
        threshold=THRESHOLD,
        adaptive_top_k=TOP_K,
    )

    fold_rows: List[Dict[str, object]] = []
    confusion_all: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "exp_id": EXP_ID,
        "candidate_pool_size": len(candidate_pool),
        "selected_top30": top_features,
        "selected_new_evt": [f for f in top_features if f in NEW_EVT_FEATURES],
        "selected_existing_evt": [f for f in top_features if f in EXISTING_EVT_FEATURES],
        "selection_basis": (
            "Global static SHAP ranking aggregated across all outer training "
            "folds (same method as S2_StaticShap30). Candidate pool is the "
            "full daily parquet plus the 6 new Step-2 inspired event features."
        ),
    }

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

        x_train = train_df[top_features].copy()
        x_test = test_df[top_features].copy()
        y_train_cont = train_df[TARGET].values
        y_test_cont = test_df[TARGET].values
        y_train_cls = label_3class(y_train_cont, threshold=THRESHOLD)
        y_test_cls = label_3class(y_test_cont, threshold=THRESHOLD)

        y_pred_cls = fit_predict(exp, x_train, y_train_cls, x_test)

        metrics = offset_metrics(y_test_cont, y_test_cls, y_pred_cls)
        maj = baseline_majority(y_train_cls, len(y_test_cls))
        mom = label_3class(test_df["td_vs_xfn_5d"].values, threshold=THRESHOLD)
        maj_metrics = offset_metrics(y_test_cont, y_test_cls, maj)
        mom_metrics = offset_metrics(y_test_cont, y_test_cls, mom)
        calibration = full_row_calibration(y_test_cls, y_pred_cls)
        confusion_all.extend(confusion_rows(EXP_ID, fold_id, y_test_cls, y_pred_cls))

        fold_rows.append(
            {
                "exp_id": EXP_ID,
                "fold": fold_id,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_count": len(top_features),
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
        "n_folds": int(len(folds_df)),
        "candidate_pool_size": len(candidate_pool),
        "top_k": TOP_K,
        "threshold": THRESHOLD,
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

    return {
        "folds_df": folds_df,
        "summary": summary,
        "manifest": manifest,
        "stats_df": stats_df,
        "confusion_df": pd.DataFrame(confusion_all),
    }


def _load_row(path: Path, exp_id: str) -> pd.Series:
    df = pd.read_csv(path)
    row = df.loc[df["exp_id"] == exp_id]
    return row.iloc[0] if not row.empty else pd.Series(dtype="float64")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = run_experiment()

    folds_path = OUT_DIR / "step2_full_staticshap_fold_metrics.csv"
    summary_path = OUT_DIR / "step2_full_staticshap_summary.json"
    manifest_path = OUT_DIR / "step2_full_staticshap_feature_manifest.json"
    stats_path = OUT_DIR / "step2_full_staticshap_stable_core_stats.csv"
    confusion_path = OUT_DIR / "step2_full_staticshap_confusion.csv"

    out["folds_df"].to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(out["summary"], indent=2))
    manifest_path.write_text(json.dumps(out["manifest"], indent=2))
    out["stats_df"].to_csv(stats_path, index=False)
    out["confusion_df"].to_csv(confusion_path, index=False)

    round2_path = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/round2_summary.csv"
    r07 = _load_row(round2_path, "S2_Reduced")
    shap30 = _load_row(round2_path, "S2_StaticShap30")

    step2_48_path = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_48/step2_48_summary.json"
    boot_path = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_48_bootstrap/step2_48_bootstrap_summary.json"
    step2_48f = json.loads(step2_48_path.read_text()) if step2_48_path.exists() else {}
    step2_48_boot = json.loads(boot_path.read_text()) if boot_path.exists() else {}

    summary = out["summary"]

    print(f"\n=== STEP2_Full_StaticShap30 — global static SHAP, full pool + Step-2 evt_* ===")
    print(f"pool size         : {summary['candidate_pool_size']}")
    print(f"threshold         : {summary['threshold']:.4f} (+/- 0.30%)")
    print(f"folds evaluated   : {summary['n_folds']}")
    print()

    headers = [
        "STEP2_Full_StaticShap30",
        "S2_StaticShap30",
        "STEP2_48F_Boot30",
        "STEP2_48F",
        "R07",
    ]
    print(
        f"{'metric':<28} "
        f"{headers[0]:>26} {headers[1]:>17} {headers[2]:>18} "
        f"{headers[3]:>11} {headers[4]:>8}"
    )
    print("-" * 116)

    def row(name: str, key: str) -> None:
        v_full = summary.get(key, float("nan"))
        v_shap = float(shap30[key]) if not shap30.empty and key in shap30 else float("nan")
        v_boot = float(step2_48_boot.get(key, float("nan"))) if step2_48_boot else float("nan")
        v_48 = float(step2_48f.get(key, float("nan"))) if step2_48f else float("nan")
        v_r07 = float(r07[key]) if not r07.empty and key in r07 else float("nan")
        print(
            f"{name:<28} {v_full:>26.4f} {v_shap:>17.4f} "
            f"{v_boot:>18.4f} {v_48:>11.4f} {v_r07:>8.4f}"
        )

    row("mean_accuracy", "mean_accuracy")
    row("mean_macro_f1", "mean_macro_f1")
    row("mean_active_sign_acc", "mean_active_sign_acc")
    row("mean_active_coverage", "mean_active_coverage")
    row("mean_active_precision_pos", "mean_active_precision_pos")
    row("mean_active_precision_neg", "mean_active_precision_neg")
    print()
    print(f"uplift vs majority : {summary['uplift_vs_majority_pp']:+.2f} pp")
    print(f"uplift vs momentum : {summary['uplift_vs_momentum_pp']:+.2f} pp")

    print(f"\nTop-30 selected features (in global SHAP rank order):")
    for i, feat in enumerate(out["manifest"]["selected_top30"], start=1):
        tag = ""
        if feat in NEW_EVT_FEATURES:
            tag = "  [NEW Step-2]"
        elif feat in EXISTING_EVT_FEATURES:
            tag = "  [existing evt_*]"
        print(f"  {i:>2}. {feat}{tag}")
    print(
        f"\nNew Step-2 features chosen: "
        f"{out['manifest']['selected_new_evt']}"
    )
    print(f"Existing evt_* chosen     : {out['manifest']['selected_existing_evt']}")
    print()
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {summary_path}")
    print(f"Saved -> {manifest_path}")
    print(f"Saved -> {stats_path}")
    print(f"Saved -> {confusion_path}")


if __name__ == "__main__":
    main()
