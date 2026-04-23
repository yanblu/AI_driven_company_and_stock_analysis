"""
Leakage-clean within-training-only SHAP stability feature selection.

For each outer fold k:
1. Take only that fold's training window (data on or before the training cutoff).
2. Inside that training window, run a short expanding-window walk-forward with
   multiple inner cutoffs.
3. For each inner cutoff, fit a shallow LightGBM on the inner training slice and
   rank features by SHAP on the inner validation slice.
4. Keep the fold-local stable feature list: features that appear in the top-M
   (default 30) in a majority of inner windows, capped at MAX_FEATURES.
5. Retrain on the full fold training window using only the fold-local stable
   feature list and evaluate on that fold's test window.

Outputs are written under redesign_single_stock/data/improvements/ and never
overwrite existing redesign artifacts.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_EXP_PARENT = Path(__file__).resolve().parents[2]  # model_experiments/
if str(_EXP_PARENT) not in sys.path:
    sys.path.insert(0, str(_EXP_PARENT))

from src.models.walk_forward_config import E10_FOLDS
from redesign_single_stock.src.run_redesign_experiments import (
    Experiment,
    baseline_majority,
    confusion_rows,
    feature_set_full,
    fit_predict,
    full_row_calibration,
    label_3class,
    load_dataset,
    multiclass_shap_importance,
    offset_metrics,
)


OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/improvements"
ARTIFACT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/artifacts"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.003
MAX_FEATURES = 30
INNER_TOPM = 30
INNER_SPLIT_FRACTIONS = [0.60, 0.70, 0.80, 0.90]
MAJORITY_HITS = 3
EXP_ID = "IMP_ShapStability30"
TITLE = "Within-training SHAP stability top 30"

BASELINE_EXP_ID = "S2_StaticShap30"
BASELINE_SUMMARY_PATH = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/round2_summary.csv"


def _fit_selector(x_fit: pd.DataFrame, y_fit_enc: np.ndarray) -> lgb.LGBMClassifier:
    selector = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=8,
        min_child_samples=30,
        feature_fraction=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    selector.fit(x_fit, y_fit_enc)
    return selector


def derive_fold_local_stable_features(
    train_df: pd.DataFrame, candidate_pool: List[str]
) -> Tuple[List[str], pd.DataFrame]:
    label_map = {-1: 0, 0: 1, 1: 2}
    inner_rows: List[Dict[str, object]] = []
    n_inner_used = 0

    for frac in INNER_SPLIT_FRACTIONS:
        split_idx = int(len(train_df) * frac)
        split_idx = min(max(split_idx, 80), len(train_df) - 40)
        fit_df = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]
        if len(val_df) < 20 or len(fit_df) < 80:
            continue

        x_fit = fit_df[candidate_pool]
        x_val = val_df[candidate_pool]
        y_fit_cls = label_3class(fit_df[TARGET].values, threshold=THRESHOLD)
        y_fit_enc = np.array([label_map[v] for v in y_fit_cls], dtype=int)
        if len(np.unique(y_fit_enc)) < 2:
            continue

        selector = _fit_selector(x_fit, y_fit_enc)
        importance = multiclass_shap_importance(selector, x_val)
        ranked = importance.index.tolist()
        n_inner_used += 1

        for rank, feature in enumerate(ranked, start=1):
            inner_rows.append(
                {
                    "inner_split_fraction": frac,
                    "feature": feature,
                    "rank": rank,
                    "importance": float(importance[feature]),
                    "in_inner_topm": rank <= INNER_TOPM,
                }
            )

    if n_inner_used == 0:
        raise RuntimeError("No inner windows produced SHAP rankings.")

    fold_stats_df = pd.DataFrame(inner_rows)
    agg = (
        fold_stats_df.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            mean_rank=("rank", "mean"),
            hits_in_topm=("in_inner_topm", "sum"),
            n_inner_windows=("in_inner_topm", "count"),
        )
    )
    majority_needed = min(MAJORITY_HITS, n_inner_used)
    agg["stable"] = agg["hits_in_topm"] >= majority_needed
    agg = agg.sort_values(
        ["stable", "hits_in_topm", "mean_importance", "mean_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    stable_features = agg.loc[agg["stable"], "feature"].tolist()
    if len(stable_features) >= MAX_FEATURES:
        selected = stable_features[:MAX_FEATURES]
    else:
        fillers = [f for f in agg["feature"].tolist() if f not in stable_features]
        selected = stable_features + fillers[: MAX_FEATURES - len(stable_features)]

    agg["selected"] = agg["feature"].isin(selected)
    agg["n_inner_windows_used"] = n_inner_used
    agg["majority_needed"] = majority_needed
    return selected, agg


def run_experiment() -> Dict[str, object]:
    df = load_dataset()
    candidate_pool = feature_set_full(df)
    exp = Experiment(
        exp_id=EXP_ID,
        stage="improvements",
        title=TITLE,
        target=TARGET,
        feature_mode="adaptive",
        model_kind="lgbm_multiclass",
        threshold=THRESHOLD,
        adaptive_top_k=MAX_FEATURES,
    )

    fold_rows: List[Dict[str, object]] = []
    confusion_matrix_rows: List[Dict[str, object]] = []
    feature_manifest: Dict[str, Dict[str, object]] = {EXP_ID: {}}
    stability_stats_rows: List[Dict[str, object]] = []

    for fold_id, train_end, test_start, test_end in E10_FOLDS:
        train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[TARGET].notna()
        test_mask = (
            (df["date"] >= pd.Timestamp(test_start))
            & (df["date"] <= pd.Timestamp(test_end))
            & df[TARGET].notna()
        )
        train_df = (
            df.loc[train_mask, ["date", TARGET] + candidate_pool]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )
        test_df = df.loc[test_mask].copy()

        selected_features, agg = derive_fold_local_stable_features(train_df, candidate_pool)
        agg_rows = agg.to_dict(orient="records")
        for row in agg_rows:
            row["fold"] = fold_id
            stability_stats_rows.append(row)

        feature_manifest[EXP_ID][fold_id] = {
            "feature_mode": "shap_stability_training_only",
            "selected_features": selected_features,
            "selection_basis": (
                "Within-training-only SHAP stability: features that appear in the top "
                f"{INNER_TOPM} in at least {MAJORITY_HITS} of {len(INNER_SPLIT_FRACTIONS)} inner expanding windows; "
                "if fewer than 30 stable, remaining slots filled by mean SHAP importance."
            ),
            "inner_split_fractions": INNER_SPLIT_FRACTIONS,
            "candidate_pool_size": len(candidate_pool),
            "n_stable_features": int(agg["stable"].sum()),
        }

        x_train = train_df[selected_features].copy()
        x_test = test_df[selected_features].copy()
        y_train_cont = train_df[TARGET].values
        y_test_cont = test_df[TARGET].values
        y_train_cls = label_3class(y_train_cont, threshold=THRESHOLD)
        y_test_cls = label_3class(y_test_cont, threshold=THRESHOLD)

        y_pred_cls = fit_predict(exp, x_train, y_train_cls, x_test)

        model_metrics = offset_metrics(y_test_cont, y_test_cls, y_pred_cls)
        majority_pred = baseline_majority(y_train_cls, len(y_test_cls))
        momentum_pred = label_3class(x_test["td_vs_xfn_5d"].values, threshold=THRESHOLD)
        majority_metrics = offset_metrics(y_test_cont, y_test_cls, majority_pred)
        momentum_metrics = offset_metrics(y_test_cont, y_test_cls, momentum_pred)
        calibration = full_row_calibration(y_test_cls, y_pred_cls)
        confusion_matrix_rows.extend(confusion_rows(EXP_ID, fold_id, y_test_cls, y_pred_cls))

        fold_rows.append(
            {
                "stage": exp.stage,
                "exp_id": exp.exp_id,
                "title": exp.title,
                "target": exp.target,
                "feature_mode": "shap_stability_training_only",
                "model_kind": exp.model_kind,
                "threshold": exp.threshold,
                "feature_count": len(selected_features),
                "fold": fold_id,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "adaptive_top_k": exp.adaptive_top_k,
                "accuracy": model_metrics["accuracy"],
                "macro_f1": model_metrics["macro_f1"],
                "active_sign_acc": model_metrics["active_sign_acc"],
                "active_coverage": model_metrics["active_coverage"],
                "majority_accuracy": majority_metrics["accuracy"],
                "majority_macro_f1": majority_metrics["macro_f1"],
                "momentum_accuracy": momentum_metrics["accuracy"],
                "momentum_macro_f1": momentum_metrics["macro_f1"],
                **calibration,
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    summary_df = (
        folds_df.groupby(
            ["stage", "exp_id", "title", "target", "model_kind", "feature_mode", "threshold", "feature_count", "adaptive_top_k"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_active_sign_acc=("active_sign_acc", "mean"),
            mean_active_coverage=("active_coverage", "mean"),
            mean_majority_accuracy=("majority_accuracy", "mean"),
            mean_momentum_accuracy=("momentum_accuracy", "mean"),
            mean_actual_neg_share=("actual_neg_share", "mean"),
            mean_actual_neu_share=("actual_neu_share", "mean"),
            mean_actual_pos_share=("actual_pos_share", "mean"),
            mean_pred_neg_share=("pred_neg_share", "mean"),
            mean_pred_neu_share=("pred_neu_share", "mean"),
            mean_pred_pos_share=("pred_pos_share", "mean"),
            mean_active_precision_pos=("active_precision_pos", "mean"),
            mean_active_precision_neg=("active_precision_neg", "mean"),
            mean_predicted_max_share=("predicted_max_share", "mean"),
        )
        .reset_index(drop=True)
    )
    summary_df["uplift_vs_majority_pp"] = 100 * (
        summary_df["mean_accuracy"] - summary_df["mean_majority_accuracy"]
    )
    summary_df["uplift_vs_momentum_pp"] = 100 * (
        summary_df["mean_accuracy"] - summary_df["mean_momentum_accuracy"]
    )
    summary_df["acceptable_candidate"] = (
        (summary_df["uplift_vs_majority_pp"] > 2.0)
        & (summary_df["uplift_vs_momentum_pp"] > 1.0)
        & (summary_df["mean_macro_f1"] > 0.34)
        & (summary_df["mean_active_sign_acc"] > 0.50)
    )

    class_balance_df = folds_df[
        [
            "stage",
            "exp_id",
            "title",
            "fold",
            "actual_neg_share",
            "actual_neu_share",
            "actual_pos_share",
            "pred_neg_share",
            "pred_neu_share",
            "pred_pos_share",
            "active_precision_pos",
            "active_precision_neg",
        ]
    ].copy()

    return {
        "summary_df": summary_df,
        "folds_df": folds_df,
        "class_balance_df": class_balance_df,
        "confusion_df": pd.DataFrame(confusion_matrix_rows),
        "feature_manifest": feature_manifest,
        "stability_stats_df": pd.DataFrame(stability_stats_rows),
    }


def _load_baseline_metrics() -> Dict[str, float]:
    if not BASELINE_SUMMARY_PATH.exists():
        return {}
    baseline_df = pd.read_csv(BASELINE_SUMMARY_PATH)
    row = baseline_df.loc[baseline_df["exp_id"] == BASELINE_EXP_ID]
    if row.empty:
        return {}
    record = row.iloc[0]
    return {
        "mean_accuracy": float(record["mean_accuracy"]),
        "mean_macro_f1": float(record["mean_macro_f1"]),
        "mean_active_sign_acc": float(record["mean_active_sign_acc"]),
        "mean_active_coverage": float(record["mean_active_coverage"]),
    }


def _beats_baseline(candidate: Dict[str, float], baseline: Dict[str, float]) -> bool:
    if not baseline:
        return False
    return (
        candidate["mean_active_sign_acc"] > baseline["mean_active_sign_acc"]
        and candidate["mean_macro_f1"] >= baseline["mean_macro_f1"] - 0.01
        and candidate["mean_active_coverage"] >= 0.70
    )


def save_new_locked_artifact(selected_features_per_fold: Dict[str, List[str]]) -> Dict[str, object]:
    """Train a production-style artifact using the last fold's selected feature list.

    Using the last (most recent) fold respects time ordering: the feature list
    was chosen without ever seeing test data, and the final model is trained on
    all labeled data we have today.
    """
    df = load_dataset()
    fold_ids = list(selected_features_per_fold.keys())
    last_fold_features = list(selected_features_per_fold[fold_ids[-1]])

    train_df = df.loc[df[TARGET].notna(), ["date", TARGET] + last_fold_features].copy()
    x_train = train_df[last_fold_features]
    y_train_cls = label_3class(train_df[TARGET].values, threshold=THRESHOLD)

    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map = {v: k for k, v in label_map.items()}
    y_train_enc = np.array([label_map[v] for v in y_train_cls], dtype=int)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=8,
        min_child_samples=40,
        feature_fraction=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(x_train, y_train_enc)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(date.today())
    model_path = ARTIFACT_DIR / f"shap-stability30-best-{stamp}.pkl"
    meta_path = ARTIFACT_DIR / f"shap-stability30-best-{stamp}.json"

    payload = {
        "model": model,
        "features": last_fold_features,
        "target": TARGET,
        "threshold": THRESHOLD,
        "label_map": label_map,
        "inverse_label_map": inv_map,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    metadata = {
        "artifact_id": f"shap-stability30-best-{stamp}",
        "source_experiment": EXP_ID,
        "description": (
            "Leakage-clean SHAP stability top-30 redesign candidate: "
            "TD 5-day excess return vs XFN, 3-class, within-training-only "
            "fold-local SHAP stability selection, shallow LightGBM multiclass."
        ),
        "target": TARGET,
        "threshold": THRESHOLD,
        "features": last_fold_features,
        "feature_selection_method": (
            "For each outer fold, SHAP stability selection using inner expanding windows "
            "on the training data only. "
            f"Top {INNER_TOPM} features per inner window; kept features appearing in at least "
            f"{MAJORITY_HITS} of {len(INNER_SPLIT_FRACTIONS)} inner windows. "
            "Locked artifact uses the last fold's selected feature list trained on all labeled data."
        ),
        "train_rows": int(len(train_df)),
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "class_distribution": {
            "underperform": float((y_train_cls == -1).mean()),
            "neutral": float((y_train_cls == 0).mean()),
            "outperform": float((y_train_cls == 1).mean()),
        },
        "artifact_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    return {"model_path": str(model_path), "meta_path": str(meta_path)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = run_experiment()

    summary_path = OUT_DIR / "shap_stability30_summary.csv"
    folds_path = OUT_DIR / "shap_stability30_fold_metrics.csv"
    class_balance_path = OUT_DIR / "shap_stability30_class_balance.csv"
    confusion_path = OUT_DIR / "shap_stability30_confusion_matrices.csv"
    manifest_path = OUT_DIR / "shap_stability30_feature_manifest.json"
    stability_stats_path = OUT_DIR / "shap_stability30_inner_stats.csv"

    outputs["summary_df"].to_csv(summary_path, index=False)
    outputs["folds_df"].to_csv(folds_path, index=False)
    outputs["class_balance_df"].to_csv(class_balance_path, index=False)
    outputs["confusion_df"].to_csv(confusion_path, index=False)
    manifest_path.write_text(json.dumps(outputs["feature_manifest"], indent=2))
    outputs["stability_stats_df"].to_csv(stability_stats_path, index=False)

    baseline = _load_baseline_metrics()
    candidate = outputs["summary_df"].iloc[0].to_dict()
    candidate_metrics = {
        "mean_accuracy": float(candidate["mean_accuracy"]),
        "mean_macro_f1": float(candidate["mean_macro_f1"]),
        "mean_active_sign_acc": float(candidate["mean_active_sign_acc"]),
        "mean_active_coverage": float(candidate["mean_active_coverage"]),
    }

    print(outputs["summary_df"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nBaseline S2_StaticShap30 metrics: {baseline}")
    print(f"Candidate IMP_ShapStability30 metrics: {candidate_metrics}")

    beats = _beats_baseline(candidate_metrics, baseline)
    decision_payload = {
        "candidate": candidate_metrics,
        "baseline_s2_staticshap30": baseline,
        "beats_baseline": bool(beats),
        "criteria": (
            "beats_baseline = (candidate active_sign_acc > baseline) AND "
            "(candidate macro_f1 >= baseline - 0.01) AND (candidate active_coverage >= 0.70)."
        ),
    }
    (OUT_DIR / "shap_stability30_decision.json").write_text(
        json.dumps(decision_payload, indent=2)
    )

    if beats:
        fold_to_features = {
            fold: manifest["selected_features"]
            for fold, manifest in outputs["feature_manifest"][EXP_ID].items()
        }
        artifact_info = save_new_locked_artifact(fold_to_features)
        print(f"\nNew leakage-clean SHAP-stability artifact saved: {artifact_info['model_path']}")
        print(f"Metadata saved: {artifact_info['meta_path']}")
    else:
        print("\nCandidate does not beat S2_StaticShap30 on business metrics; no new artifact saved.")

    print(f"\nSaved -> {summary_path}")
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {class_balance_path}")
    print(f"Saved -> {confusion_path}")
    print(f"Saved -> {manifest_path}")
    print(f"Saved -> {stability_stats_path}")


if __name__ == "__main__":
    main()
