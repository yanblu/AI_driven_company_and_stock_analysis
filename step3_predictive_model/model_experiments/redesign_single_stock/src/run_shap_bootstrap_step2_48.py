"""
Bootstrap SHAP voting on the 48-feature Step-2 grounded pool.

Identical methodology to `run_shap_bootstrap_experiment.py`
(`IMP_ShapBootstrap30`), but the candidate pool is the 48 features defined
in `nlp_event_feature_design.md` Section 9 instead of the full 180+ column
daily parquet. Expectation: fewer redundant `_ffill`/`_delta`/`_surprise`
duplicates should let votes concentrate, so the selected ~30 should carry
the `v4`/`v5` gains of `STEP2_48F` while dropping the features that hurt
`v1`/`v6`/`v7`.

Outputs: `redesign_single_stock/data/step2_48_bootstrap/`.
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
    fit_predict,
    full_row_calibration,
    label_3class,
    offset_metrics,
)
from redesign_single_stock.src.run_shap_bootstrap_experiment import (  # noqa: E402
    BASE_SEED,
    FREQ_THRESHOLD,
    MAX_FEATURES,
    N_BOOTSTRAPS,
    SUBTRAIN_FRACTION,
    TOPM_PER_BOOTSTRAP,
    bootstrap_vote_features,
)
from redesign_single_stock.src.run_step2_48feature_experiment import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURES_48,
    NEW_EVT_FEATURES,
    load_dataset_step2,
)
from src.models.walk_forward_config import E10_FOLDS  # noqa: E402


OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_48_bootstrap"
TARGET = "target_excess_xfn_5d"
THRESHOLD = 0.0030
EXP_ID = "STEP2_48F_Boot30"
TITLE = "Bootstrap SHAP voting top-30 on the 48-feature Step-2 pool"


def run_experiment() -> Dict[str, object]:
    df = load_dataset_step2()

    missing = [c for c in FEATURES_48 if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing features in dataset: {missing}")

    exp = Experiment(
        exp_id=EXP_ID,
        stage="step2_48_bootstrap",
        title=TITLE,
        target=TARGET,
        feature_mode="bootstrap_shap_voting",
        model_kind="lgbm_multiclass",
        threshold=THRESHOLD,
        adaptive_top_k=MAX_FEATURES,
    )

    fold_rows: List[Dict[str, object]] = []
    confusion_all: List[Dict[str, object]] = []
    stats_all: List[Dict[str, object]] = []
    manifest: Dict[str, Dict[str, object]] = {EXP_ID: {}}

    feature_group_lookup: Dict[str, str] = {
        feat: group for group, feats in FEATURE_GROUPS.items() for feat in feats
    }

    for i, (fold_id, train_end, test_start, test_end) in enumerate(E10_FOLDS):
        train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[TARGET].notna()
        test_mask = (
            (df["date"] >= pd.Timestamp(test_start))
            & (df["date"] <= pd.Timestamp(test_end))
            & df[TARGET].notna()
        )
        train_df = (
            df.loc[train_mask, ["date", TARGET] + FEATURES_48]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )
        test_df = df.loc[test_mask].copy()

        selected, stats_df = bootstrap_vote_features(
            train_df=train_df,
            candidate_pool=FEATURES_48,
            fold_seed=BASE_SEED + i,
        )
        stats_df["fold"] = fold_id
        stats_df["feature_group"] = stats_df["feature"].map(feature_group_lookup)
        stats_all.extend(stats_df.to_dict(orient="records"))

        manifest[EXP_ID][fold_id] = {
            "feature_mode": "bootstrap_shap_voting_48pool",
            "selected_features": selected,
            "n_selected": len(selected),
            "n_stable_features": int(stats_df["stable"].sum()),
            "selection_basis": (
                f"Bootstrap SHAP voting inside the fold's training window: "
                f"{N_BOOTSTRAPS} random {int(SUBTRAIN_FRACTION * 100)}/"
                f"{int((1 - SUBTRAIN_FRACTION) * 100)} splits, top-"
                f"{TOPM_PER_BOOTSTRAP} per split, kept if freq >= "
                f"{int(FREQ_THRESHOLD * 100)}%, filled to {MAX_FEATURES} by "
                "mean SHAP importance. Candidate pool = 48-feature Step-2 pool."
            ),
            "candidate_pool_size": len(FEATURES_48),
            "new_evt_features_selected": [
                f for f in NEW_EVT_FEATURES if f in selected
            ],
        }

        x_train = train_df[selected].copy()
        x_test = test_df[selected].copy()
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
                "feature_count": len(selected),
                "threshold": THRESHOLD,
                "n_stable_features": int(stats_df["stable"].sum()),
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
        "candidate_pool_size": len(FEATURES_48),
        "threshold": THRESHOLD,
        "max_features": MAX_FEATURES,
        "mean_accuracy": float(folds_df["accuracy"].mean()),
        "mean_macro_f1": float(folds_df["macro_f1"].mean()),
        "mean_active_sign_acc": float(folds_df["active_sign_acc"].mean()),
        "mean_active_coverage": float(folds_df["active_coverage"].mean()),
        "mean_majority_accuracy": float(folds_df["majority_accuracy"].mean()),
        "mean_momentum_accuracy": float(folds_df["momentum_accuracy"].mean()),
        "mean_active_precision_pos": float(folds_df["active_precision_pos"].mean()),
        "mean_active_precision_neg": float(folds_df["active_precision_neg"].mean()),
        "mean_predicted_max_share": float(folds_df["predicted_max_share"].mean()),
        "mean_feature_count": float(folds_df["feature_count"].mean()),
        "mean_n_stable_features": float(folds_df["n_stable_features"].mean()),
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
        "confusion_df": pd.DataFrame(confusion_all),
        "stats_df": pd.DataFrame(stats_all),
        "feature_manifest": manifest,
    }


def _compare_row(df: pd.DataFrame, exp_id: str) -> pd.Series:
    row = df.loc[df["exp_id"] == exp_id]
    if row.empty:
        return pd.Series(dtype="float64")
    return row.iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = run_experiment()

    folds_path = OUT_DIR / "step2_48_bootstrap_fold_metrics.csv"
    summary_path = OUT_DIR / "step2_48_bootstrap_summary.json"
    manifest_path = OUT_DIR / "step2_48_bootstrap_feature_manifest.json"
    confusion_path = OUT_DIR / "step2_48_bootstrap_confusion.csv"
    stats_path = OUT_DIR / "step2_48_bootstrap_vote_stats.csv"

    out["folds_df"].to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(out["summary"], indent=2))
    manifest_path.write_text(json.dumps(out["feature_manifest"], indent=2))
    out["confusion_df"].to_csv(confusion_path, index=False)
    out["stats_df"].to_csv(stats_path, index=False)

    round2 = pd.read_csv(ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/round2_summary.csv")
    r07 = _compare_row(round2, "S2_Reduced")
    shap30 = _compare_row(round2, "S2_StaticShap30")

    step2_summary_path = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/step2_48/step2_48_summary.json"
    step2_48f = json.loads(step2_summary_path.read_text()) if step2_summary_path.exists() else {}

    summary = out["summary"]
    print("\n=== Bootstrap SHAP on 48-feature Step-2 pool ===")
    print(f"pool size             : {summary['candidate_pool_size']}")
    print(f"threshold             : {summary['threshold']:.4f} (+/- 0.30%)")
    print(f"folds evaluated       : {summary['n_folds']}")
    print(f"mean feature_count    : {summary['mean_feature_count']:.1f}")
    print(f"mean stable features  : {summary['mean_n_stable_features']:.1f}")
    print()
    headers = ["STEP2_48F_Boot30", "STEP2_48F", "R07", "S2_StaticShap30 (leaky)"]
    print(
        f"{'metric':<28} {headers[0]:>18} {headers[1]:>12} "
        f"{headers[2]:>10} {headers[3]:>24}"
    )
    print("-" * 100)

    def row(name: str, key: str) -> None:
        val_boot = summary.get(key, float("nan"))
        val_step2 = float(step2_48f.get(key, float("nan"))) if step2_48f else float("nan")
        val_r07 = float(r07[key]) if not r07.empty and key in r07 else float("nan")
        val_shap = float(shap30[key]) if not shap30.empty and key in shap30 else float("nan")
        print(
            f"{name:<28} {val_boot:>18.4f} {val_step2:>12.4f} "
            f"{val_r07:>10.4f} {val_shap:>24.4f}"
        )

    row("mean_accuracy", "mean_accuracy")
    row("mean_macro_f1", "mean_macro_f1")
    row("mean_active_sign_acc", "mean_active_sign_acc")
    row("mean_active_coverage", "mean_active_coverage")
    row("mean_active_precision_pos", "mean_active_precision_pos")
    row("mean_active_precision_neg", "mean_active_precision_neg")
    print()
    print(f"uplift vs majority    : {summary['uplift_vs_majority_pp']:+.2f} pp")
    print(f"uplift vs momentum    : {summary['uplift_vs_momentum_pp']:+.2f} pp")
    print()
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {summary_path}")
    print(f"Saved -> {manifest_path}")
    print(f"Saved -> {confusion_path}")
    print(f"Saved -> {stats_path}")


if __name__ == "__main__":
    main()
