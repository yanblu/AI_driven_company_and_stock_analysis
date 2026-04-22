"""
Train and save the current best redesign model artifact (R07).

Artifact is kept separate from the locked raw-return model so nothing gets
overwritten. The saved payload includes:
  - trained LightGBM multiclass model
  - feature list
  - target definition
  - class threshold
  - label encoding
  - train date range
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_EXP_PARENT = Path(__file__).resolve().parents[2]  # model_experiments/
if str(_EXP_PARENT) not in sys.path:
    sys.path.insert(0, str(_EXP_PARENT))

from redesign_single_stock.src.run_redesign_experiments import (
    REDUCED_FEATURES,
    label_3class,
    load_dataset,
)


OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data/artifacts"
THRESHOLD = 0.003
TARGET = "target_excess_xfn_5d"


def main() -> None:
    df = load_dataset()
    train_df = df.loc[df[TARGET].notna(), ["date", TARGET] + REDUCED_FEATURES].copy()

    x_train = train_df[REDUCED_FEATURES]
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
    model_path = OUT_DIR / f"r07-current-best-{stamp}.pkl"
    meta_path = OUT_DIR / f"r07-current-best-{stamp}.json"

    payload = {
        "model": model,
        "features": REDUCED_FEATURES,
        "target": TARGET,
        "threshold": THRESHOLD,
        "label_map": label_map,
        "inverse_label_map": inv_map,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    metadata = {
        "artifact_id": f"r07-current-best-{stamp}",
        "source_experiment": "R07",
        "description": "Current best redesign model: TD 5-day excess return vs XFN, 3-class, reduced event-style features, shallow LightGBM multiclass.",
        "target": TARGET,
        "threshold": THRESHOLD,
        "features": REDUCED_FEATURES,
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

    print(f"Saved model artifact -> {model_path}")
    print(f"Saved metadata      -> {meta_path}")


if __name__ == "__main__":
    main()
