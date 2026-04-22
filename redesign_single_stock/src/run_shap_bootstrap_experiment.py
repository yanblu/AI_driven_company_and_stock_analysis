"""
Bootstrap SHAP voting feature selection (simpler stability selection).

For each outer fold k:
1. Take only that fold's training window (data on or before train_end[k]).
2. Repeat N times:
   - Randomly split the training window into a sub-train and a sub-test.
   - Fit a shallow LightGBM on sub-train.
   - Compute SHAP importance on sub-test.
   - Record that bootstrap's top-M features.
3. Selection frequency per feature = (# bootstraps it appeared in top-M) / N.
4. Keep features with selection frequency >= FREQ_THRESHOLD, capped at 30.
   If fewer than 30 clear winners, fill remaining slots by mean SHAP importance.
5. Retrain on the full fold training window using only those 30 features,
   evaluate on the outer test window.

This is the Meinshausen-Buhlmann style stability selection:
- random subsamples inside the fold training window
- aggregate by vote frequency, not by raw SHAP magnitude
- no information from later folds or from the outer test window ever enters
  the feature list for fold k

Note on the 5-day forward target: a random sub-train/sub-test split can put
two rows with overlapping forward-return windows in different halves. This is
only used for feature ranking (SHAP importance), not for any performance claim,
so the overlap does not contaminate the outer walk-forward evaluation.

Outputs are written under redesign_single_stock/data/improvements/.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


OUT_DIR = ROOT / "redesign_single_stock/data/improvements"
ARTIFACT_DIR = ROOT / "redesign_single_stock/data/artifacts"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.003

MAX_FEATURES = 30
TOPM_PER_BOOTSTRAP = 30
N_BOOTSTRAPS = 20
SUBTRAIN_FRACTION = 0.70
FREQ_THRESHOLD = 0.50
BASE_SEED = 42

EXP_ID = "IMP_ShapBootstrap30"
TITLE = "Bootstrap SHAP voting top 30"

BASELINE_EXP_ID = "S2_StaticShap30"
BASELINE_SUMMARY_PATH = ROOT / "redesign_single_stock/data/round2_summary.csv"


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


def bootstrap_vote_features(
    train_df: pd.DataFrame,
    candidate_pool: List[str],
    fold_seed: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Random sub-train / sub-test SHAP voting inside a single fold's training window."""
    label_map = {-1: 0, 0: 1, 1: 2}
    rng = np.random.default_rng(fold_seed)
    n_rows = len(train_df)
    indices = np.arange(n_rows)

    top_hits = pd.Series(0, index=candidate_pool, dtype=int)
    importance_sum = pd.Series(0.0, index=candidate_pool, dtype=float)
    n_bootstraps_used = 0

    for b in range(N_BOOTSTRAPS):
        shuffled = rng.permutation(indices)
        cut = int(n_rows * SUBTRAIN_FRACTION)
        sub_train_idx = shuffled[:cut]
        sub_test_idx = shuffled[cut:]
        if len(sub_train_idx) < 80 or len(sub_test_idx) < 40:
            continue

        sub_train = train_df.iloc[sub_train_idx]
        sub_test = train_df.iloc[sub_test_idx]

        y_fit_cls = label_3class(sub_train[TARGET].values, threshold=THRESHOLD)
        y_fit_enc = np.array([label_map[v] for v in y_fit_cls], dtype=int)
        if len(np.unique(y_fit_enc)) < 2:
            continue

        x_fit = sub_train[candidate_pool]
        x_val = sub_test[candidate_pool]

        selector = _fit_selector(x_fit, y_fit_enc)
        importance = multiclass_shap_importance(selector, x_val)

        importance_sum = importance_sum.add(importance, fill_value=0.0)
        top_features = importance.index[:TOPM_PER_BOOTSTRAP]
        top_hits.loc[top_features] += 1
        n_bootstraps_used += 1

    if n_bootstraps_used == 0:
        raise RuntimeError("No bootstrap samples were successfully fit.")

    selection_frequency = top_hits / n_bootstraps_used
    mean_importance = importance_sum / n_bootstraps_used

    stats_df = pd.DataFrame(
        {
            "feature": candidate_pool,
            "selection_frequency": selection_frequency.values,
            "top_hits": top_hits.values,
            "n_bootstraps_used": n_bootstraps_used,
            "mean_importance": mean_importance.reindex(candidate_pool).values,
        }
    )
    stats_df["stable"] = stats_df["selection_frequency"] >= FREQ_THRESHOLD
    stats_df = stats_df.sort_values(
        ["stable", "selection_frequency", "mean_importance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    stable_features = stats_df.loc[stats_df["stable"], "feature"].tolist()
    if len(stable_features) >= MAX_FEATURES:
        selected = stable_features[:MAX_FEATURES]
    else:
        fillers = [f for f in stats_df["feature"].tolist() if f not in stable_features]
        selected = stable_features + fillers[: MAX_FEATURES - len(stable_features)]

    stats_df["selected"] = stats_df["feature"].isin(selected)
    return selected, stats_df


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
    stats_rows: List[Dict[str, object]] = []

    for i, (fold_id, train_end, test_start, test_end) in enumerate(E10_FOLDS):
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

        selected_features, stats_df = bootstrap_vote_features(
            train_df=train_df,
            candidate_pool=candidate_pool,
            fold_seed=BASE_SEED + i,
        )
        stats_df["fold"] = fold_id
        stats_rows.extend(stats_df.to_dict(orient="records"))

        feature_manifest[EXP_ID][fold_id] = {
            "feature_mode": "bootstrap_shap_voting",
            "selected_features": selected_features,
            "selection_basis": (
                f"Bootstrap SHAP voting inside the fold's training window: {N_BOOTSTRAPS} random "
                f"{int(SUBTRAIN_FRACTION * 100)}/{int((1 - SUBTRAIN_FRACTION) * 100)} splits. "
                f"Features kept if in top-{TOPM_PER_BOOTSTRAP} in at least {int(FREQ_THRESHOLD * 100)}% "
                "of bootstraps; remaining slots filled by mean SHAP importance to reach 30."
            ),
            "n_bootstraps": N_BOOTSTRAPS,
            "subtrain_fraction": SUBTRAIN_FRACTION,
            "freq_threshold": FREQ_THRESHOLD,
            "candidate_pool_size": len(candidate_pool),
            "n_stable_features": int(stats_df["stable"].sum()),
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
                "feature_mode": "bootstrap_shap_voting",
                "model_kind": exp.model_kind,
                "threshold": exp.threshold,
                "feature_count": len(selected_features),
                "fold": fold_id,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "adaptive_top_k": exp.adaptive_top_k,
                "n_stable_features": int(stats_df["stable"].sum()),
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
        "bootstrap_stats_df": pd.DataFrame(stats_rows),
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
    """Train a production-style artifact using the last fold's selected feature list."""
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
    model_path = ARTIFACT_DIR / f"shap-bootstrap30-best-{stamp}.pkl"
    meta_path = ARTIFACT_DIR / f"shap-bootstrap30-best-{stamp}.json"

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
        "artifact_id": f"shap-bootstrap30-best-{stamp}",
        "source_experiment": EXP_ID,
        "description": (
            "Leakage-clean bootstrap SHAP voting top-30 redesign candidate: "
            "TD 5-day excess return vs XFN, 3-class, bootstrap stability "
            "selection inside each fold's training window, shallow LightGBM."
        ),
        "target": TARGET,
        "threshold": THRESHOLD,
        "features": last_fold_features,
        "feature_selection_method": (
            f"For each outer fold, {N_BOOTSTRAPS} random "
            f"{int(SUBTRAIN_FRACTION * 100)}/{int((1 - SUBTRAIN_FRACTION) * 100)} splits of the "
            "fold's training window. Fit LightGBM on each sub-train, compute SHAP on the sub-test, "
            f"keep features appearing in top-{TOPM_PER_BOOTSTRAP} in at least "
            f"{int(FREQ_THRESHOLD * 100)}% of bootstraps; fill to 30 by mean importance. "
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

    summary_path = OUT_DIR / "shap_bootstrap30_summary.csv"
    folds_path = OUT_DIR / "shap_bootstrap30_fold_metrics.csv"
    class_balance_path = OUT_DIR / "shap_bootstrap30_class_balance.csv"
    confusion_path = OUT_DIR / "shap_bootstrap30_confusion_matrices.csv"
    manifest_path = OUT_DIR / "shap_bootstrap30_feature_manifest.json"
    stats_path = OUT_DIR / "shap_bootstrap30_vote_stats.csv"

    outputs["summary_df"].to_csv(summary_path, index=False)
    outputs["folds_df"].to_csv(folds_path, index=False)
    outputs["class_balance_df"].to_csv(class_balance_path, index=False)
    outputs["confusion_df"].to_csv(confusion_path, index=False)
    manifest_path.write_text(json.dumps(outputs["feature_manifest"], indent=2))
    outputs["bootstrap_stats_df"].to_csv(stats_path, index=False)

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
    print(f"Candidate IMP_ShapBootstrap30 metrics: {candidate_metrics}")

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
    (OUT_DIR / "shap_bootstrap30_decision.json").write_text(
        json.dumps(decision_payload, indent=2)
    )

    if beats:
        fold_to_features = {
            fold: manifest["selected_features"]
            for fold, manifest in outputs["feature_manifest"][EXP_ID].items()
        }
        artifact_info = save_new_locked_artifact(fold_to_features)
        print(f"\nNew leakage-clean bootstrap SHAP artifact saved: {artifact_info['model_path']}")
        print(f"Metadata saved: {artifact_info['meta_path']}")
    else:
        print("\nCandidate does not beat S2_StaticShap30 on business metrics; no new artifact saved.")

    print(f"\nSaved -> {summary_path}")
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {class_balance_path}")
    print(f"Saved -> {confusion_path}")
    print(f"Saved -> {manifest_path}")
    print(f"Saved -> {stats_path}")


if __name__ == "__main__":
    main()
