"""
Leakage-clean fold-local SHAP top-30 experiment.

For each outer fold:
1. fit a selector on that fold's training window only
2. compute SHAP importance using only data inside that training window
3. keep the fold-local top-30 feature list
4. retrain the same model on the full training window using only those features
5. evaluate on the outer test fold

Outputs are written under redesign_single_stock/data/improvements/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

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
    offset_metrics,
    rank_adaptive_features,
)


OUT_DIR = ROOT / "redesign_single_stock/data/improvements"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.003
TOP_K = 30
EXP_ID = "IMP_FoldLocalShap30"
TITLE = "Fold-local SHAP top 30 features"


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
        adaptive_top_k=TOP_K,
    )

    fold_rows: List[Dict[str, object]] = []
    confusion_matrix_rows: List[Dict[str, object]] = []
    feature_manifest: Dict[str, Dict[str, object]] = {}

    for fold_id, train_end, test_start, test_end in E10_FOLDS:
        train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[TARGET].notna()
        test_mask = (
            (df["date"] >= pd.Timestamp(test_start))
            & (df["date"] <= pd.Timestamp(test_end))
            & df[TARGET].notna()
        )

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        ranked_features, _, selection_meta = rank_adaptive_features(
            train_df=train_df[["date", TARGET] + candidate_pool].copy(),
            candidate_features=candidate_pool,
            target=TARGET,
            threshold=THRESHOLD,
        )
        selected_features = ranked_features[:TOP_K]
        feature_manifest.setdefault(EXP_ID, {})[fold_id] = {
            "feature_mode": "fold_local_shap",
            "selected_features": selected_features,
            "selection_basis": "Fold-local SHAP ranking computed from training data only.",
            "ranked_candidates_top25": ranked_features[:25],
            "candidate_pool_size": selection_meta["candidate_pool_size"],
            "selection_fit_rows": selection_meta["fit_rows"],
            "selection_validation_rows": selection_meta["validation_rows"],
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
                "feature_mode": "fold_local_shap",
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
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = run_experiment()

    summary_path = OUT_DIR / "fold_local_shap30_summary.csv"
    folds_path = OUT_DIR / "fold_local_shap30_fold_metrics.csv"
    class_balance_path = OUT_DIR / "fold_local_shap30_class_balance.csv"
    confusion_path = OUT_DIR / "fold_local_shap30_confusion_matrices.csv"
    manifest_path = OUT_DIR / "fold_local_shap30_feature_manifest.json"

    outputs["summary_df"].to_csv(summary_path, index=False)
    outputs["folds_df"].to_csv(folds_path, index=False)
    outputs["class_balance_df"].to_csv(class_balance_path, index=False)
    outputs["confusion_df"].to_csv(confusion_path, index=False)
    manifest_path.write_text(json.dumps(outputs["feature_manifest"], indent=2))

    print(outputs["summary_df"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved -> {summary_path}")
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {class_balance_path}")
    print(f"Saved -> {confusion_path}")
    print(f"Saved -> {manifest_path}")


if __name__ == "__main__":
    main()
