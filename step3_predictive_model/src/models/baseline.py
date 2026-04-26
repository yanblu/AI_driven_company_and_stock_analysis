"""
Train two Random Forest baselines on the daily feature matrix.

  rf_option_a : train Feb 2021 → Dec 31 2024  |  test Jan 2025 → Apr 2026
  rf_option_b : train Feb 2021 → Aug 22 2024  |  test Aug 23 2024 → Apr 2026
                (cutoff = day after FY2024Q3 call; crisis + recovery in test)

Both models use identical hyperparameters and median imputation so results
are directly comparable.

Outputs saved to data/processed/models/:
  rf_option_a.joblib   rf_option_a_results.json
  rf_option_c.joblib   rf_option_c_results.json
  shap_option_a.parquet
  shap_option_c.parquet
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # project root
FEATURES_PATH = ROOT / "step1_data_collection/data/features/model_features_daily.parquet"
OUT_DIR = ROOT / "step3_predictive_model/model_experiments_archive/archive_baseline_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Split cutoffs
# ---------------------------------------------------------------------------
CUTOFF_A = pd.Timestamp("2024-12-31")   # Option A: train through Dec 2024
CUTOFF_C = pd.Timestamp("2024-08-22")   # Option C: train through FY2024Q3 call date

TARGET = "td_return_5d_fwd"

# Random Forest hyperparameters (identical for both models)
RF_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame, cutoff: pd.Timestamp):
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"date", TARGET}
    return [c for c in df.columns if c not in drop]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    return {"r2": round(r2, 4), "mae": round(mae, 5), "rmse": round(rmse, 5),
            "directional_accuracy": round(direction_acc, 4)}


def train_and_evaluate(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    label: str,
) -> dict:
    train, test = split(df, cutoff)
    feature_cols = get_feature_cols(df)

    X_train = train[feature_cols]
    y_train = train[TARGET].values
    X_test = test[feature_cols]
    y_test = test[TARGET].values

    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"  Train: {train['date'].min().date()} → {train['date'].max().date()} ({len(train)} rows)")
    print(f"  Test:  {test['date'].min().date()} → {test['date'].max().date()} ({len(test)} rows)")
    print(f"  Features: {len(feature_cols)}")

    # All missing values are resolved in the feature builder (zero-fill at source).
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train.values, y_train)

    y_train_pred = rf_model.predict(X_train.values)
    y_test_pred = rf_model.predict(X_test.values)

    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    print(f"  Train metrics: {train_metrics}")
    print(f"  Test  metrics: {test_metrics}")

    # SHAP values on the test set
    print("  Computing SHAP values on test set...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test.values)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "date", test["date"].values)
    shap_df.to_parquet(OUT_DIR / f"shap_{label}.parquet", index=False)

    # Global feature importance from SHAP (mean |SHAP|)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    top20 = mean_abs_shap.head(20).round(6).to_dict()
    print(f"  Top 5 SHAP features: {dict(list(top20.items())[:5])}")

    # Save model
    dump(rf_model, OUT_DIR / f"{label}.joblib")

    # Save results
    results = {
        "label": label,
        "cutoff": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())],
        "n_features": len(feature_cols),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "top20_shap_features": top20,
    }
    with open(OUT_DIR / f"{label}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also save test predictions for notebook plotting
    pred_df = test[["date", TARGET]].copy()
    pred_df["predicted"] = y_test_pred
    pred_df.to_parquet(OUT_DIR / f"{label}_predictions.parquet", index=False)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns")

    results_a = train_and_evaluate(df, CUTOFF_A, "rf_option_a")
    results_c = train_and_evaluate(df, CUTOFF_C, "rf_option_c")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for label, res in [("Option A (train→Dec 2024)", results_a), ("Option C (train→Aug 2024)", results_c)]:
        m = res["test_metrics"]
        print(f"\n{label}")
        print(f"  Test R²:               {m['r2']}")
        print(f"  Test MAE:              {m['mae']}")
        print(f"  Test Directional Acc:  {m['directional_accuracy']}")

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
