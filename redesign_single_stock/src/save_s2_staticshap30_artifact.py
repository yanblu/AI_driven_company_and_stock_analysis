"""
Train and save the best SHAP-selected redesign model artifact (S2_StaticShap30).

This keeps the SHAP-based candidate preserved separately from the current
overall-best manual redesign artifact.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redesign_single_stock.src.run_redesign_experiments import label_3class, load_dataset


OUT_DIR = ROOT / "redesign_single_stock/data/artifacts"
STATS_PATH = ROOT / "redesign_single_stock/data/round2_stable_core_stats.csv"
SUMMARY_PATH = ROOT / "redesign_single_stock/data/round2_summary.csv"
THRESHOLD = 0.003
TARGET = "target_excess_xfn_5d"
TOP_K = 30
SOURCE_EXPERIMENT = "S2_StaticShap30"


def load_shap_feature_list() -> list[str]:
    stats_df = pd.read_csv(STATS_PATH)
    return stats_df.head(TOP_K)["feature"].tolist()


def load_summary_metrics() -> dict[str, float]:
    summary_df = pd.read_csv(SUMMARY_PATH)
    row = summary_df.loc[summary_df["exp_id"] == SOURCE_EXPERIMENT]
    if row.empty:
        return {}
    record = row.iloc[0]
    return {
        "mean_accuracy": float(record["mean_accuracy"]),
        "mean_macro_f1": float(record["mean_macro_f1"]),
        "mean_active_sign_acc": float(record["mean_active_sign_acc"]),
        "mean_active_coverage": float(record["mean_active_coverage"]),
    }


def main() -> None:
    features = load_shap_feature_list()
    df = load_dataset()
    train_df = df.loc[df[TARGET].notna(), ["date", TARGET] + features].copy()

    x_train = train_df[features]
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(date.today())
    model_path = OUT_DIR / f"s2-staticshap30-best-shap-{stamp}.pkl"
    meta_path = OUT_DIR / f"s2-staticshap30-best-shap-{stamp}.json"

    payload = {
        "model": model,
        "features": features,
        "target": TARGET,
        "threshold": THRESHOLD,
        "label_map": label_map,
        "inverse_label_map": inv_map,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    metadata = {
        "artifact_id": f"s2-staticshap30-best-shap-{stamp}",
        "source_experiment": SOURCE_EXPERIMENT,
        "description": "Best SHAP-selected redesign candidate: TD 5-day excess return vs XFN, 3-class, static SHAP-selected top 30 features, shallow LightGBM multiclass.",
        "target": TARGET,
        "threshold": THRESHOLD,
        "features": features,
        "feature_selection_method": "Top 30 features from the global SHAP stability ranking aggregated across training folds.",
        "train_rows": int(len(train_df)),
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "class_distribution": {
            "underperform": float((y_train_cls == -1).mean()),
            "neutral": float((y_train_cls == 0).mean()),
            "outperform": float((y_train_cls == 1).mean()),
        },
        "round2_summary_metrics": load_summary_metrics(),
        "artifact_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model artifact -> {model_path}")
    print(f"Saved metadata      -> {meta_path}")


if __name__ == "__main__":
    main()
