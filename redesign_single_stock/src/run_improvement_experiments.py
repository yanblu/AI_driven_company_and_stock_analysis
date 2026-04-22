"""
Post-hoc improvement experiments benchmarked against S2_StaticShap30.

Three families of improvements, each stacked on top of the same base LightGBM
trained with the top-30 SHAP-stable features at ±30 bps threshold:

  1. Probability calibration  — Platt (sigmoid) and isotonic regression
  2. Confidence threshold     — abstain to neutral when max class prob < cutoff
  3. Regime filter            — gate to neutral when td_volatility_20d exceeds
                                 a percentile computed on the training fold
  4. Combinations             — calibration + threshold / regime

No saved model artifacts are read or modified.
All outputs go to redesign_single_stock/data/improvements/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redesign_single_stock.src.run_redesign_experiments import (
    full_row_calibration,
    label_3class,
    load_dataset,
    offset_metrics,
)
from src.models.walk_forward_config import E10_FOLDS, STRIDE_EVAL

# ── Constants ──────────────────────────────────────────────────────────────
STATS_PATH = ROOT / "redesign_single_stock/data/round2_stable_core_stats.csv"
OUT_DIR    = ROOT / "redesign_single_stock/data/improvements"

TARGET     = "target_excess_xfn_5d"
THRESHOLD  = 0.003
TOP_K      = 30
STRIDE     = STRIDE_EVAL
OFFSETS    = list(range(STRIDE))

LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
INV_MAP    = {v: k for k, v in LABEL_MAP.items()}
NEUTRAL_ENC = LABEL_MAP[0]            # class index 1 = neutral / abstain

VOL_COL    = "td_volatility_20d"      # used for regime gate (not in SHAP-30)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_shap_features() -> List[str]:
    stats_df = pd.read_csv(STATS_PATH)
    return stats_df.head(TOP_K)["feature"].tolist()


def fit_base(x_fit: pd.DataFrame, y_fit: np.ndarray) -> lgb.LGBMClassifier:
    """Same hyperparameters as S2_StaticShap30."""
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
    model.fit(x_fit, y_fit)
    return model


def decode(pred_enc: np.ndarray) -> np.ndarray:
    return np.array([INV_MAP[int(v)] for v in pred_enc], dtype=int)


def platt_calibrate(
    proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray
) -> np.ndarray:
    """One-vs-rest Platt sigmoid calibration. Returns renormalised probabilities."""
    n_classes = proba_cal.shape[1]
    out = np.zeros_like(proba_test)
    for c in range(n_classes):
        y_bin = (y_cal == c).astype(int)
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            out[:, c] = proba_test[:, c]
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(proba_cal[:, c].reshape(-1, 1), y_bin)
        out[:, c] = lr.predict_proba(proba_test[:, c].reshape(-1, 1))[:, 1]
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.where(row_sum == 0, 1.0, row_sum)


def isotonic_calibrate(
    proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray
) -> np.ndarray:
    """One-vs-rest isotonic regression calibration. Returns renormalised probabilities."""
    n_classes = proba_cal.shape[1]
    out = np.zeros_like(proba_test)
    for c in range(n_classes):
        y_bin = (y_cal == c).astype(float)
        if y_bin.sum() < 3:
            out[:, c] = proba_test[:, c]
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba_cal[:, c], y_bin)
        out[:, c] = ir.predict(proba_test[:, c])
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.where(row_sum == 0, 1.0, row_sum)


def apply_conf_threshold(proba: np.ndarray, thr: float) -> np.ndarray:
    """Abstain (→ neutral) when max class probability is below thr."""
    pred = np.argmax(proba, axis=1).copy()
    pred[proba.max(axis=1) < thr] = NEUTRAL_ENC
    return pred


def apply_regime_gate(
    pred_enc: np.ndarray,
    test_vol: np.ndarray,
    vol_cutoff: float,
) -> np.ndarray:
    """Force neutral when test-day volatility exceeds the training-fold cutoff."""
    out = pred_enc.copy()
    high_vol = np.isfinite(test_vol) & (test_vol > vol_cutoff)
    out[high_vol] = NEUTRAL_ENC
    return out


# ── Experiment spec ────────────────────────────────────────────────────────

@dataclass
class ImprovExperiment:
    exp_id: str
    description: str
    calibration: str  = "none"      # "none" | "platt" | "isotonic"
    conf_thr: float   = 0.0         # 0.0 = disabled
    regime_pct: float = 0.0         # 0.0 = disabled; e.g. 0.75 → p75 of train vol


def build_experiments() -> List[ImprovExperiment]:
    return [
        # ── Baseline (reproduces S2_StaticShap30 exactly) ────────────────
        ImprovExperiment("Baseline",            "S2_StaticShap30 baseline"),

        # ── Calibration only ─────────────────────────────────────────────
        ImprovExperiment("Platt",               "Platt sigmoid calibration",
                         calibration="platt"),
        ImprovExperiment("Isotonic",            "Isotonic regression calibration",
                         calibration="isotonic"),

        # ── Confidence threshold only ─────────────────────────────────────
        ImprovExperiment("Conf55",              "Confidence gate 0.55",
                         conf_thr=0.55),
        ImprovExperiment("Conf60",              "Confidence gate 0.60",
                         conf_thr=0.60),
        ImprovExperiment("Conf65",              "Confidence gate 0.65",
                         conf_thr=0.65),
        ImprovExperiment("Conf70",              "Confidence gate 0.70",
                         conf_thr=0.70),

        # ── Regime filter only ────────────────────────────────────────────
        ImprovExperiment("Regime75",            "Regime gate vol > p75",
                         regime_pct=0.75),
        ImprovExperiment("Regime80",            "Regime gate vol > p80",
                         regime_pct=0.80),
        ImprovExperiment("Regime85",            "Regime gate vol > p85",
                         regime_pct=0.85),

        # ── Calibration + confidence ──────────────────────────────────────
        ImprovExperiment("Platt_Conf60",        "Platt + conf gate 0.60",
                         calibration="platt",    conf_thr=0.60),
        ImprovExperiment("Platt_Conf65",        "Platt + conf gate 0.65",
                         calibration="platt",    conf_thr=0.65),
        ImprovExperiment("Isotonic_Conf60",     "Isotonic + conf gate 0.60",
                         calibration="isotonic", conf_thr=0.60),
        ImprovExperiment("Isotonic_Conf65",     "Isotonic + conf gate 0.65",
                         calibration="isotonic", conf_thr=0.65),

        # ── Calibration + regime ──────────────────────────────────────────
        ImprovExperiment("Platt_Regime75",      "Platt + regime gate p75",
                         calibration="platt",    regime_pct=0.75),
        ImprovExperiment("Platt_Regime80",      "Platt + regime gate p80",
                         calibration="platt",    regime_pct=0.80),

        # ── Triple combos ─────────────────────────────────────────────────
        ImprovExperiment("Platt_Conf65_R75",    "Platt + conf 0.65 + regime p75",
                         calibration="platt",    conf_thr=0.65, regime_pct=0.75),
        ImprovExperiment("Platt_Conf65_R80",    "Platt + conf 0.65 + regime p80",
                         calibration="platt",    conf_thr=0.65, regime_pct=0.80),
        ImprovExperiment("Isotonic_Conf65_R75", "Isotonic + conf 0.65 + regime p75",
                         calibration="isotonic", conf_thr=0.65, regime_pct=0.75),
    ]


# ── Fold runner ────────────────────────────────────────────────────────────

def run_fold(
    exp: ImprovExperiment,
    features: List[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    y_train_cont = train_df[TARGET].values
    y_test_cont  = test_df[TARGET].values
    y_train_cls  = label_3class(y_train_cont, threshold=THRESHOLD)
    y_test_cls   = label_3class(y_test_cont,  threshold=THRESHOLD)
    y_train_enc  = np.array([LABEL_MAP[v] for v in y_train_cls], dtype=int)

    x_train = train_df[features].copy()
    x_test  = test_df[features].copy()

    # ── Fit model (with or without calibration split) ────────────────────
    if exp.calibration in ("platt", "isotonic"):
        # Chronological 80/20 split: model on first 80%, calibrator on last 20%
        n = len(x_train)
        split = max(min(int(n * 0.80), n - 30), 60)

        x_fit, y_fit = x_train.iloc[:split], y_train_enc[:split]
        x_cal, y_cal = x_train.iloc[split:],  y_train_enc[split:]

        base       = fit_base(x_fit, y_fit)
        proba_cal  = base.predict_proba(x_cal)
        proba_test = base.predict_proba(x_test)

        if exp.calibration == "platt":
            proba = platt_calibrate(proba_cal, y_cal, proba_test)
        else:
            proba = isotonic_calibrate(proba_cal, y_cal, proba_test)
    else:
        base  = fit_base(x_train, y_train_enc)
        proba = base.predict_proba(x_test)

    # ── Confidence threshold ─────────────────────────────────────────────
    if exp.conf_thr > 0.0:
        pred_enc = apply_conf_threshold(proba, exp.conf_thr)
    else:
        pred_enc = np.argmax(proba, axis=1)

    # ── Regime filter ─────────────────────────────────────────────────────
    if exp.regime_pct > 0.0 and VOL_COL in train_df.columns:
        train_vol = train_df[VOL_COL].dropna().values
        if len(train_vol) > 0:
            vol_cutoff = float(np.nanpercentile(train_vol, exp.regime_pct * 100))
            test_vol = test_df[VOL_COL].fillna(0.0).values
            pred_enc = apply_regime_gate(pred_enc, test_vol, vol_cutoff)

    y_pred_cls = decode(pred_enc)

    metrics = offset_metrics(y_test_cont, y_test_cls, y_pred_cls)
    cal     = full_row_calibration(y_test_cls, y_pred_cls)
    return {**metrics, **cal}


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading dataset …")
    df       = load_dataset()
    features = load_shap_features()
    print(f"  Features: {len(features)} SHAP-stable columns")

    experiments = build_experiments()
    fold_rows: List[Dict[str, object]] = []

    total = len(experiments) * len(E10_FOLDS)
    done  = 0

    for exp in experiments:
        print(f"\n[{exp.exp_id}]  {exp.description}")
        for fold_id, train_end, test_start, test_end in E10_FOLDS:
            train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[TARGET].notna()
            test_mask  = (
                (df["date"] >= pd.Timestamp(test_start))
                & (df["date"] <= pd.Timestamp(test_end))
                & df[TARGET].notna()
            )
            train_df = df.loc[train_mask].copy()
            test_df  = df.loc[test_mask].copy()

            result = run_fold(exp, features, train_df, test_df)
            done  += 1

            fold_rows.append({
                "exp_id":        exp.exp_id,
                "description":   exp.description,
                "calibration":   exp.calibration,
                "conf_thr":      exp.conf_thr,
                "regime_pct":    exp.regime_pct,
                "fold":          fold_id,
                "train_rows":    len(train_df),
                "test_rows":     len(test_df),
                **result,
            })
            print(f"  {fold_id}: sign_acc={result['active_sign_acc']:.3f}  "
                  f"cov={result['active_coverage']:.3f}  "
                  f"pos_prec={result.get('active_precision_pos', float('nan')):.3f}  "
                  f"neg_prec={result.get('active_precision_neg', float('nan')):.3f}  "
                  f"({done}/{total})")

    folds_df = pd.DataFrame(fold_rows)

    # ── Summary aggregation ───────────────────────────────────────────────
    group_cols = ["exp_id", "description", "calibration", "conf_thr", "regime_pct"]
    summary_df = (
        folds_df.groupby(group_cols, as_index=False)
        .agg(
            mean_accuracy           = ("accuracy",             "mean"),
            mean_macro_f1           = ("macro_f1",             "mean"),
            mean_active_sign_acc    = ("active_sign_acc",      "mean"),
            mean_active_coverage    = ("active_coverage",      "mean"),
            mean_precision_pos      = ("active_precision_pos", "mean"),
            mean_precision_neg      = ("active_precision_neg", "mean"),
            mean_pred_neg_share     = ("pred_neg_share",       "mean"),
            mean_pred_neu_share     = ("pred_neu_share",       "mean"),
            mean_pred_pos_share     = ("pred_pos_share",       "mean"),
        )
        .reset_index(drop=True)
        .sort_values("mean_active_sign_acc", ascending=False)
        .reset_index(drop=True)
    )

    # Compute delta vs baseline for each metric
    baseline_row = summary_df.loc[summary_df["exp_id"] == "Baseline"].iloc[0]
    for col in ["mean_accuracy", "mean_macro_f1", "mean_active_sign_acc",
                "mean_active_coverage", "mean_precision_pos", "mean_precision_neg"]:
        summary_df[f"delta_{col}"] = summary_df[col] - baseline_row[col]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_path   = OUT_DIR / "improvement_fold_metrics.csv"
    summary_path = OUT_DIR / "improvement_summary.csv"
    folds_df.to_csv(folds_path,   index=False)
    summary_df.to_csv(summary_path, index=False)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("IMPROVEMENT EXPERIMENT SUMMARY  (sorted by active sign accuracy)")
    print("=" * 110)
    display = [
        "exp_id",
        "mean_active_sign_acc",
        "delta_mean_active_sign_acc",
        "mean_active_coverage",
        "mean_precision_pos",
        "mean_precision_neg",
        "mean_accuracy",
        "mean_macro_f1",
    ]
    print(summary_df[display].to_string(index=False, float_format=lambda x: f"{x:+.4f}" if x < 0 or (x > 0 and x < 1) else f"{x:.4f}"))
    print(f"\nSaved → {folds_path}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
