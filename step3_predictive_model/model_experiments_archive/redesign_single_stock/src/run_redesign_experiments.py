"""
Round-2 redesign experiments for the single-stock TD vs XFN classifier.

This script keeps the original locked raw-return model untouched and writes all
round-2 outputs to the dedicated redesign folder under redesign_single_stock/data/.

Round-2 focus:
1. schema-first documentation of the full candidate feature universe
2. conservative band sweep around the current R07 setup
3. hybrid feature selection: fixed core + adaptive per-retrain SHAP features
4. LightGBM vs XGBoost model comparison on the best label/feature setup
5. calibration diagnostics for predicted vs realized class balance
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.walk_forward_config import E10_FOLDS, STRIDE_EVAL

FEAT_PATH = ROOT / "data/processed/features/model_features_daily.parquet"
PRICES_PATH = ROOT / "data/raw/prices"
OUT_DIR = ROOT / "step3_predictive_model/model_experiments/redesign_single_stock/data"

ROUND2_PREFIX = "round2"
TARGET_THRESHOLD = 0.005  # +/- 50 bps for active under/outperform labels
CLASS_ORDER = [-1, 0, 1]
STRIDE = STRIDE_EVAL
OFFSETS = list(range(STRIDE))


def label_3class(values: np.ndarray, threshold: float = TARGET_THRESHOLD) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    out[values > threshold] = 1
    out[values < -threshold] = -1
    return out


def decay_from_days(days: pd.Series, halflife_days: float = 20.0) -> pd.Series:
    clipped = days.clip(lower=0).fillna(999.0)
    return np.exp(-clipped / halflife_days)


def load_price_returns(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICES_PATH / f"{ticker}.parquet")[["date", "adj_close"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df[f"{ticker.lower()}_fwd5"] = df["adj_close"].pct_change(5).shift(-5)
    return df[["date", f"{ticker.lower()}_fwd5"]]


def load_dataset() -> pd.DataFrame:
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

    engineered = pd.DataFrame(index=df.index)
    engineered["target_excess_xfn_5d"] = df["td_fwd5"] - df["xfn_fwd5"]
    engineered["target_excess_tsx_5d"] = df["td_fwd5"] - df["tsx_fwd5"]

    call_decay = decay_from_days(df["days_since_call"])
    news_decay = np.exp(-df["days_since_last_news"].fillna(999.0).clip(lower=0) / 15.0)

    # Event-style NLP compression: current quarter information is strongest near the call date.
    engineered["evt_ceo_tone"] = df["transcript_ceo_prep_sentiment_mean_ffill"] * call_decay
    engineered["evt_cfo_tone"] = df["transcript_cfo_prep_sentiment_mean_ffill"] * call_decay
    engineered["evt_framing_gap"] = df["framing_gap_ffill"] * call_decay
    engineered["evt_aml_pressure"] = (
        df["topic_regulatory_AML_share_ffill"] * (1.0 - df["topic_regulatory_AML_sentiment_ffill"])
    ) * call_decay
    engineered["evt_aml_shift"] = (
        df["topic_regulatory_AML_share_delta"].abs()
        + df["topic_regulatory_AML_sentiment_delta"].abs()
    ) * call_decay
    engineered["evt_guidance_strength"] = (
        df["topic_guidance_share_ffill"] * df["topic_guidance_sentiment_ffill"]
    ) * call_decay
    engineered["evt_guidance_shift"] = (
        df["topic_guidance_share_delta"] + df["topic_guidance_sentiment_delta"]
    ) * call_decay
    engineered["evt_macro_topic"] = (
        df["topic_macro_outlook_share_ffill"] * (df["topic_macro_outlook_sentiment_ffill"] - 0.5)
    ) * call_decay
    engineered["evt_topic_entropy"] = df["topic_entropy_ffill"] * call_decay
    engineered["evt_news_tone"] = df["news_sent_mean_30d"] * news_decay
    engineered["evt_news_flow"] = np.log1p(df["news_count_30d"]) * news_decay

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

    return pd.concat([df, engineered], axis=1).copy()


BASE_REDUCED_FEATURES = [
    "td_return_5d",
    "td_return_20d",
    "td_volatility_20d",
    "td_vs_xfn_5d",
    "td_vs_tsx_5d",
    "td_corr_pv_20d",
    "td_volume_change_20d",
    "td_dist_52w_high",
    "yield_curve_slope",
    "yield_10y_level",
    "vix_volatility_20d",
    "dxy_level",
    "gold_level",
    "fx_usdcad_level",
    "news_sent_mean_30d",
    "news_count_30d",
    "days_since_last_news",
    "days_since_call",
    "is_earnings_week",
]

EVENT_FEATURES = [
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
]

REDUCED_FEATURES = BASE_REDUCED_FEATURES + EVENT_FEATURES
AML_EVENT_FEATURES = ["evt_aml_pressure", "evt_aml_shift"]
MANUAL_NO_AML_FEATURES = [f for f in REDUCED_FEATURES if f not in AML_EVENT_FEATURES]
TOPIC_BASKET_FEATURES = ["evt_broad_topic_signal", "evt_broad_topic_shift"]
MANUAL_TOPIC_BASKET_FEATURES = MANUAL_NO_AML_FEATURES + TOPIC_BASKET_FEATURES
FIXED_CORE_SIZE = 15
NLP_PCA_COMPONENTS = 10
STABLE_CORE_TOP10_CUTOFF = 10
STABLE_CORE_TOP20_CUTOFF = 20
TARGET_COLS = {"date", "td_return_5d_fwd", "td_fwd5", "xfn_fwd5", "tsx_fwd5", "target_excess_xfn_5d", "target_excess_tsx_5d"}

EVENT_DESCRIPTIONS = {
    "evt_ceo_tone": "CEO prepared-remarks tone decayed by days since the last earnings call.",
    "evt_cfo_tone": "CFO prepared-remarks tone decayed by days since the last earnings call.",
    "evt_framing_gap": "Gap between management tone and filed report tone, decayed after the call.",
    "evt_aml_pressure": "Regulatory AML topic pressure combining topic share and negative sentiment, decayed after the call.",
    "evt_aml_shift": "Absolute quarter-over-quarter AML narrative shift in topic share and sentiment, decayed after the call.",
    "evt_guidance_strength": "Positive guidance strength in the latest call, decayed after the call.",
    "evt_guidance_shift": "Quarter-over-quarter guidance shift, decayed after the call.",
    "evt_macro_topic": "Macro topic emphasis and tone in the latest call, decayed after the call.",
    "evt_topic_entropy": "Topic breadth from the latest call, decayed after the call.",
    "evt_news_tone": "Recent 30-day news tone decayed by days since the most recent article.",
    "evt_news_flow": "Recent 30-day news intensity decayed by days since the most recent article.",
    "evt_broad_topic_signal": "Average decayed signal across non-AML business topics such as credit quality, capital, wealth, retail, and NIM.",
    "evt_broad_topic_shift": "Average decayed quarter-over-quarter shift across a broader non-AML topic basket.",
}


def feature_set_full(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in TARGET_COLS]


def nlp_block_features(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in feature_set_full(df)
        if c.startswith(("transcript_", "report_sentiment_", "topic_", "framing_gap"))
        or c.startswith("news_sentiment_mean_")
    ]


def baseline_majority(y_train_cls: np.ndarray, size: int) -> np.ndarray:
    values, counts = np.unique(y_train_cls, return_counts=True)
    return np.full(size, values[np.argmax(counts)], dtype=int)


def baseline_relative_momentum(x_test: pd.DataFrame, target_name: str) -> np.ndarray:
    anchor = "td_vs_xfn_5d" if "xfn" in target_name else "td_vs_tsx_5d"
    return label_3class(x_test[anchor].values, threshold=TARGET_THRESHOLD)


def offset_metrics(y_true_cont: np.ndarray, y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> Dict[str, float]:
    accs: List[float] = []
    f1s: List[float] = []
    active_accs: List[float] = []
    coverages: List[float] = []

    for offset in OFFSETS:
        idx = np.arange(offset, len(y_true_cls), STRIDE)
        yt_cls = y_true_cls[idx]
        yp_cls = y_pred_cls[idx]
        yt_cont = y_true_cont[idx]

        accs.append(float((yp_cls == yt_cls).mean()))
        f1s.append(float(f1_score(yt_cls, yp_cls, labels=CLASS_ORDER, average="macro", zero_division=0)))

        active = yp_cls != 0
        coverages.append(float(active.mean()))
        if active.any():
            active_accs.append(float((np.sign(yp_cls[active]) == np.sign(yt_cont[active])).mean()))
        else:
            active_accs.append(np.nan)

    valid_active = [x for x in active_accs if not np.isnan(x)]

    return {
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "active_sign_acc": float(np.mean(valid_active)) if valid_active else np.nan,
        "active_coverage": float(np.mean(coverages)),
    }


def full_row_calibration(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label_id, label_name in [(-1, "neg"), (0, "neu"), (1, "pos")]:
        out[f"actual_{label_name}_share"] = float((y_true_cls == label_id).mean())
        out[f"pred_{label_name}_share"] = float((y_pred_cls == label_id).mean())

    pos_mask = y_pred_cls == 1
    neg_mask = y_pred_cls == -1
    out["active_precision_pos"] = float((y_true_cls[pos_mask] == 1).mean()) if pos_mask.any() else np.nan
    out["active_precision_neg"] = float((y_true_cls[neg_mask] == -1).mean()) if neg_mask.any() else np.nan
    out["predicted_max_share"] = float(
        max(out["pred_neg_share"], out["pred_neu_share"], out["pred_pos_share"])
    )
    return out


def confusion_rows(exp_id: str, fold_id: str, y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> List[Dict[str, object]]:
    label_name = {-1: "underperform", 0: "neutral", 1: "outperform"}
    rows: List[Dict[str, object]] = []
    for actual in CLASS_ORDER:
        for predicted in CLASS_ORDER:
            rows.append(
                {
                    "exp_id": exp_id,
                    "fold": fold_id,
                    "actual_class": label_name[actual],
                    "predicted_class": label_name[predicted],
                    "count": int(((y_true_cls == actual) & (y_pred_cls == predicted)).sum()),
                }
            )
    return rows


def classify_feature_group(feature: str) -> str:
    if feature in EVENT_FEATURES:
        return "Event-style NLP"
    if feature.startswith(("transcript_", "report_", "topic_", "framing_gap")):
        return "Quarterly NLP"
    if feature.startswith(("news_",)) or feature in {"days_since_last_news", "is_news_burst"}:
        return "News"
    if feature in {"days_since_call", "is_earnings_week"}:
        return "Timing"
    if feature.startswith(("vix_", "gold_", "dxy_")):
        return "Macro fear"
    if "fx_usdcad" in feature:
        return "FX"
    if any(token in feature for token in ["rate_", "yield_", "cpi"]):
        return "Rates & Inflation"
    if (
        "volume" in feature
        or any(token in feature for token in ["corr_pv", "wvma", "cntd", "sumd", "rsv_", "beta_", "rsqr", "resi_"])
        or feature.startswith(("td_rsi", "td_bb_", "td_dist_"))
    ):
        return "Volume & Technical"
    if feature.startswith(("td_return_", "xfn_return_", "tsx_return_", "td_vs_", "td_volatility_")):
        return "Price & Relative Momentum"
    return "Other"


def feature_source(feature: str) -> str:
    group = classify_feature_group(feature)
    if group == "Event-style NLP":
        return "Engineered in redesign_single_stock/src/run_redesign_experiments.py from quarterly NLP and rolling news columns."
    if group in {"Quarterly NLP", "Timing", "News"}:
        return "Inherited from data/processed/features/model_features_daily.parquet."
    if group in {"Price & Relative Momentum", "Volume & Technical"}:
        return "Derived from data/raw/prices/*.parquet and materialized in model_features_daily.parquet."
    if group in {"Rates & Inflation", "Macro fear", "FX"}:
        return "Derived from data/raw/macro/*.parquet and materialized in model_features_daily.parquet."
    return "Inherited from model_features_daily.parquet."


def feature_construction_logic(feature: str) -> str:
    if feature in EVENT_DESCRIPTIONS:
        return EVENT_DESCRIPTIONS[feature]
    if feature.endswith("_ffill"):
        return "Quarterly NLP level forward-filled from the earnings call activation date."
    if feature.endswith("_delta"):
        return "Quarter-over-quarter NLP change activated on the earnings call date."
    if feature.endswith("_surprise"):
        return "Quarterly NLP deviation from trailing history activated on the earnings call date."
    if feature.startswith("news_sent_mean_"):
        return "Rolling average of news sentiment over the trailing window."
    if feature.startswith("news_count_"):
        return "Rolling count of news items over the trailing window."
    if feature == "days_since_last_news":
        return "Days since the latest news article available by day d."
    if feature == "days_since_call":
        return "Days since the most recent earnings call activation date."
    if feature == "is_earnings_week":
        return "Binary flag for the first 5 days after the latest earnings call."
    if feature.startswith(("td_vs_xfn_", "td_vs_tsx_")):
        return "Relative return spread between TD and its benchmark over the matching lookback window."
    if feature.startswith(("td_return_", "xfn_return_", "tsx_return_")):
        return "Backward-looking adjusted return over the named window."
    if feature.endswith("volatility_20d"):
        return "Rolling 20-day volatility or volatility proxy."
    if "volume" in feature:
        return "Backward-looking volume or participation signal."
    if any(token in feature for token in ["corr_pv", "wvma", "cntd", "sumd", "rsv_", "beta_", "rsqr", "resi_"]):
        return "Technical / Qlib-style price-volume feature carried from the daily feature matrix."
    if any(token in feature for token in ["yield_", "rate_", "cpi", "vix_", "gold_", "dxy_", "fx_usdcad"]):
        return "Backward-looking macro or market regime feature at day d."
    return "Candidate feature inherited from the daily modeling matrix."


def feature_update_frequency(feature: str) -> str:
    group = classify_feature_group(feature)
    if group == "Quarterly NLP":
        return "Quarterly snapshot projected to daily"
    if group == "Event-style NLP":
        return "Daily decay of quarterly or rolling-event signal"
    if group == "News":
        return "Daily rolling window"
    if feature == "cpi_level" or feature == "cpi_yoy_change":
        return "Monthly source forward-filled to daily"
    return "Daily"


def feature_activation_timing(feature: str) -> str:
    group = classify_feature_group(feature)
    if group == "Quarterly NLP":
        return "Available on or after the earnings call date, then forward-filled."
    if group == "Event-style NLP":
        return "Available end-of-day d using the latest call/news information with decay."
    if group == "News":
        return "Available end-of-day d using the trailing news window."
    return "Available end-of-day d."


def feature_missing_treatment(feature: str) -> str:
    if feature.endswith(("_delta", "_surprise")):
        return "Upstream -9999 sentinel is converted to NaN in the redesign loader; tree models use native NaN and linear models use median imputation."
    return "Any remaining NaN is passed natively to tree models and median-imputed for linear models."


def build_feature_schema(
    df: pd.DataFrame,
    fixed_core_features: List[str],
    stable_core_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature in feature_set_full(df):
        rows.append(
            {
                "feature_name": feature,
                "group": classify_feature_group(feature),
                "source": feature_source(feature),
                "construction_logic": feature_construction_logic(feature),
                "update_frequency": feature_update_frequency(feature),
                "activation_timing": feature_activation_timing(feature),
                "missing_value_treatment": feature_missing_treatment(feature),
                "selection_role": "fixed_core" if feature in fixed_core_features else "adaptive_candidate",
                "in_current_reduced_set": feature in REDUCED_FEATURES,
                "is_event_feature": feature in EVENT_FEATURES,
            }
        )
    schema_df = pd.DataFrame(rows)

    if not stable_core_stats_df.empty:
        stats_cols = stable_core_stats_df.rename(columns={"feature": "feature_name"})
        schema_df = schema_df.merge(
            stats_cols[
                [
                    "feature_name",
                    "stable_core_rank",
                    "mean_norm_shap",
                    "median_norm_shap",
                    "mean_rank",
                    "median_rank",
                    "top10_share",
                    "top20_share",
                    "stable_core_selected",
                ]
            ],
            on="feature_name",
            how="left",
        )

    return schema_df.sort_values(["selection_role", "group", "feature_name"]).reset_index(drop=True)


def multiclass_shap_importance(model: lgb.LGBMClassifier, x_val: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_val, check_additivity=False)

    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(v) for v in shap_values], axis=-1)
        importances = stacked.mean(axis=(0, 2))
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            importances = np.abs(shap_array).mean(axis=(0, 2))
        else:
            importances = np.abs(shap_array).mean(axis=0)
    return pd.Series(importances, index=x_val.columns).sort_values(ascending=False)


def rank_adaptive_features(
    train_df: pd.DataFrame,
    candidate_features: List[str],
    target: str,
    threshold: float,
) -> Tuple[List[str], pd.Series, Dict[str, object]]:
    split_idx = int(len(train_df) * 0.8)
    split_idx = min(max(split_idx, 80), len(train_df) - 40)

    fit_df = train_df.iloc[:split_idx].copy()
    val_df = train_df.iloc[split_idx:].copy()

    x_fit = fit_df[candidate_features]
    x_val = val_df[candidate_features]
    y_fit = label_3class(fit_df[target].values, threshold=threshold)

    label_map = {-1: 0, 0: 1, 1: 2}
    y_fit_enc = np.array([label_map[v] for v in y_fit], dtype=int)

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
    importance = multiclass_shap_importance(selector, x_val)
    ranked = importance.index.tolist()

    return ranked, importance, {
        "candidate_pool_size": len(candidate_features),
        "fit_rows": int(len(fit_df)),
        "validation_rows": int(len(val_df)),
    }


def derive_stable_core_features(
    df: pd.DataFrame,
    target: str,
    threshold: float,
    core_size: int = FIXED_CORE_SIZE,
) -> Tuple[List[str], pd.DataFrame]:
    candidate_pool = feature_set_full(df)
    fold_rows: List[Dict[str, object]] = []

    for fold_id, train_end, _, _ in E10_FOLDS:
        train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[target].notna()
        train_df = df.loc[train_mask, ["date", target] + candidate_pool].copy()

        ranked, importance, _ = rank_adaptive_features(
            train_df=train_df,
            candidate_features=candidate_pool,
            target=target,
            threshold=threshold,
        )
        norm_importance = importance / importance.sum()

        for rank, feature in enumerate(ranked, start=1):
            fold_rows.append(
                {
                    "fold": fold_id,
                    "feature": feature,
                    "shap_importance": float(importance[feature]),
                    "norm_shap_importance": float(norm_importance[feature]),
                    "rank": rank,
                    "top10_hit": rank <= STABLE_CORE_TOP10_CUTOFF,
                    "top20_hit": rank <= STABLE_CORE_TOP20_CUTOFF,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    stats_df = (
        fold_df.groupby("feature", as_index=False)
        .agg(
            mean_norm_shap=("norm_shap_importance", "mean"),
            median_norm_shap=("norm_shap_importance", "median"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            top10_share=("top10_hit", "mean"),
            top20_share=("top20_hit", "mean"),
            fold_count=("fold", "nunique"),
        )
        .sort_values(
            ["top20_share", "top10_share", "median_norm_shap", "mean_norm_shap", "mean_rank"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    stats_df["stable_core_rank"] = np.arange(1, len(stats_df) + 1)
    stable_core_features = stats_df.head(core_size)["feature"].tolist()
    stats_df["stable_core_selected"] = stats_df["feature"].isin(stable_core_features)
    return stable_core_features, stats_df


@dataclass
class Experiment:
    exp_id: str
    stage: str
    title: str
    target: str
    feature_mode: str
    model_kind: str
    threshold: float = TARGET_THRESHOLD
    adaptive_top_k: int = 0


def fit_predict(
    exp: Experiment,
    x_train: pd.DataFrame,
    y_train_cls: np.ndarray,
    x_test: pd.DataFrame,
) -> np.ndarray:
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map = {v: k for k, v in label_map.items()}

    if exp.model_kind == "logistic":
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        C=0.5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(x_train, y_train_cls)
        return pipe.predict(x_test)

    if exp.model_kind == "lgbm_multiclass":
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
        pred_enc = model.predict(x_test)
        return np.array([inv_map[int(v)] for v in pred_enc], dtype=int)

    if exp.model_kind == "xgb_multiclass":
        y_train_enc = np.array([label_map[v] for v in y_train_cls], dtype=int)
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(x_train, y_train_enc)
        pred_enc = model.predict(x_test).astype(int)
        return np.array([inv_map[int(v)] for v in pred_enc], dtype=int)

    raise ValueError(f"Unknown model kind: {exp.model_kind}")


def resolve_features(
    exp: Experiment,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    fold_id: str,
    fixed_core_features: List[str],
    stable_feature_ranking: List[str],
    selection_cache: Dict[Tuple[float, str], Dict[str, object]],
    feature_manifest: Dict[str, Dict[str, object]],
) -> List[str]:
    if exp.feature_mode == "manual_reduced":
        features = REDUCED_FEATURES
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": [f for f in features if f in fixed_core_features],
            "adaptive_features": [f for f in features if f not in fixed_core_features],
        }
        return features

    if exp.feature_mode == "manual_no_aml":
        features = MANUAL_NO_AML_FEATURES
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": [f for f in features if f in fixed_core_features],
            "adaptive_features": [f for f in features if f not in fixed_core_features],
            "ablation_note": "Manual reduced set with AML-specific event features removed.",
        }
        return features

    if exp.feature_mode == "manual_topic_basket":
        features = MANUAL_TOPIC_BASKET_FEATURES
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": [f for f in features if f in fixed_core_features],
            "adaptive_features": [f for f in features if f not in fixed_core_features],
            "ablation_note": "Manual reduced set with AML-specific features replaced by a broader non-AML topic basket.",
        }
        return features

    if exp.feature_mode == "pca_nlp":
        features = BASE_REDUCED_FEATURES + nlp_block_features(df)
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": [f for f in BASE_REDUCED_FEATURES if f in fixed_core_features],
            "adaptive_features": [],
            "selection_basis": "Raw market/news/timing block plus PCA-compressed original NLP block.",
        }
        return features

    if exp.feature_mode == "fixed_core":
        if not fixed_core_features:
            raise ValueError("Stable fixed core has not been derived yet.")
        features = fixed_core_features
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": features,
            "adaptive_features": [],
        }
        return features

    if exp.feature_mode == "static_shap":
        if not stable_feature_ranking:
            raise ValueError("Static SHAP ranking has not been derived yet.")
        features = stable_feature_ranking[: exp.adaptive_top_k]
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": [f for f in features if f in fixed_core_features],
            "adaptive_features": [],
            "selection_basis": "Global SHAP stability ranking aggregated across outer training folds.",
            "stable_feature_ranking_top25": stable_feature_ranking[:25],
        }
        return features

    if exp.feature_mode == "adaptive":
        if not fixed_core_features:
            raise ValueError("Stable fixed core has not been derived yet.")
        cache_key = (exp.threshold, fold_id)
        if cache_key not in selection_cache:
            candidate_pool = [f for f in feature_set_full(df) if f not in fixed_core_features]
            ranked, _, meta = rank_adaptive_features(
                train_df=train_df[["date", exp.target] + candidate_pool].copy(),
                candidate_features=candidate_pool,
                target=exp.target,
                threshold=exp.threshold,
            )
            selection_cache[cache_key] = {"ranked_features": ranked, **meta}

        ranked_features = selection_cache[cache_key]["ranked_features"]
        adaptive_features = ranked_features[: exp.adaptive_top_k]
        features = fixed_core_features + adaptive_features
        feature_manifest.setdefault(exp.exp_id, {})[fold_id] = {
            "feature_mode": exp.feature_mode,
            "selected_features": features,
            "fixed_core_features": fixed_core_features,
            "adaptive_features": adaptive_features,
            "ranked_adaptive_candidates": ranked_features[:25],
            "candidate_pool_size": selection_cache[cache_key]["candidate_pool_size"],
            "selection_fit_rows": selection_cache[cache_key]["fit_rows"],
            "selection_validation_rows": selection_cache[cache_key]["validation_rows"],
        }
        return features

    raise ValueError(f"Unknown feature mode: {exp.feature_mode}")


def prepare_model_inputs(
    exp: Experiment,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if exp.feature_mode != "pca_nlp":
        return train_df[features].copy(), test_df[features].copy(), {}

    raw_features = BASE_REDUCED_FEATURES
    nlp_features = [f for f in nlp_block_features(train_df) if f in train_df.columns]

    if not nlp_features:
        raise ValueError("No NLP block features found for PCA experiment.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_nlp = imputer.fit_transform(train_df[nlp_features])
    test_nlp = imputer.transform(test_df[nlp_features])
    train_nlp = scaler.fit_transform(train_nlp)
    test_nlp = scaler.transform(test_nlp)

    n_components = min(
        exp.adaptive_top_k or NLP_PCA_COMPONENTS,
        train_nlp.shape[1],
        max(1, train_nlp.shape[0] - 1),
    )
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_nlp)
    test_pca = pca.transform(test_nlp)
    pca_cols = [f"nlp_pca_{i + 1}" for i in range(n_components)]

    x_train = pd.concat(
        [
            train_df[raw_features].reset_index(drop=True),
            pd.DataFrame(train_pca, columns=pca_cols),
        ],
        axis=1,
    )
    x_test = pd.concat(
        [
            test_df[raw_features].reset_index(drop=True),
            pd.DataFrame(test_pca, columns=pca_cols),
        ],
        axis=1,
    )

    return x_train, x_test, {
        "raw_base_features": raw_features,
        "nlp_block_features": nlp_features,
        "nlp_pca_components": pca_cols,
        "pca_total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }


def summarise_experiments(folds_df: pd.DataFrame) -> pd.DataFrame:
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
    return summary_df.sort_values(
        ["mean_active_sign_acc", "mean_active_coverage", "mean_accuracy", "mean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)


def choose_stage_winner(summary_df: pd.DataFrame, stage: str) -> pd.Series:
    stage_df = summary_df[summary_df["stage"] == stage].copy()
    filtered = stage_df[stage_df["mean_active_coverage"] >= 0.70].copy()
    if filtered.empty:
        filtered = stage_df
    return filtered.sort_values(
        ["mean_active_sign_acc", "mean_active_coverage", "mean_accuracy", "mean_macro_f1"],
        ascending=False,
    ).iloc[0]


def build_round2_plan() -> List[Experiment]:
    band_thresholds = [0.0020, 0.0025, 0.0030, 0.0035]
    experiments: List[Experiment] = []

    for threshold in band_thresholds:
        band_pp = int(threshold * 10000)
        experiments.append(
            Experiment(
                exp_id=f"S1_B{band_pp:02d}",
                stage="stage1_band",
                title=f"Band sweep {threshold:.4f} with current reduced features and LightGBM",
                target="target_excess_xfn_5d",
                feature_mode="manual_reduced",
                model_kind="lgbm_multiclass",
                threshold=threshold,
            )
        )

    for threshold in [0.0020, 0.0025]:
        band_pp = int(threshold * 10000)
        experiments.append(
            Experiment(
                exp_id=f"S1_XGB_B{band_pp:02d}",
                stage="stage1_xgb_band",
                title=f"Band sweep {threshold:.4f} with current reduced features and XGBoost",
                target="target_excess_xfn_5d",
                feature_mode="manual_reduced",
                model_kind="xgb_multiclass",
                threshold=threshold,
            )
        )
    return experiments


def append_feature_stage(experiments: List[Experiment], best_threshold: float) -> None:
    experiments.extend(
        [
            Experiment(
                exp_id="S2_Reduced",
                stage="stage2_features",
                title="Current reduced 30-feature set",
                target="target_excess_xfn_5d",
                feature_mode="manual_reduced",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
            ),
            Experiment(
                exp_id="S2_NLPPCA",
                stage="stage2_features",
                title="NLP-block PCA plus raw market features",
                target="target_excess_xfn_5d",
                feature_mode="pca_nlp",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
                adaptive_top_k=NLP_PCA_COMPONENTS,
            ),
            Experiment(
                exp_id="S2_NoAML",
                stage="stage2_features",
                title="Manual reduced set without AML-specific features",
                target="target_excess_xfn_5d",
                feature_mode="manual_no_aml",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
            ),
            Experiment(
                exp_id="S2_TopicBasket",
                stage="stage2_features",
                title="Manual reduced set with broader non-AML topic basket",
                target="target_excess_xfn_5d",
                feature_mode="manual_topic_basket",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
            ),
            Experiment(
                exp_id="S2_FixedCore",
                stage="stage2_features",
                title="Stable SHAP-selected core only",
                target="target_excess_xfn_5d",
                feature_mode="fixed_core",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
            ),
            Experiment(
                exp_id="S2_StaticShap20",
                stage="stage2_features",
                title="Static SHAP-selected top 20 features",
                target="target_excess_xfn_5d",
                feature_mode="static_shap",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
                adaptive_top_k=20,
            ),
            Experiment(
                exp_id="S2_StaticShap25",
                stage="stage2_features",
                title="Static SHAP-selected top 25 features",
                target="target_excess_xfn_5d",
                feature_mode="static_shap",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
                adaptive_top_k=25,
            ),
            Experiment(
                exp_id="S2_StaticShap30",
                stage="stage2_features",
                title="Static SHAP-selected top 30 features",
                target="target_excess_xfn_5d",
                feature_mode="static_shap",
                model_kind="lgbm_multiclass",
                threshold=best_threshold,
                adaptive_top_k=30,
            ),
        ]
    )


def append_model_stage(experiments: List[Experiment], best_threshold: float, best_feature_mode: str, adaptive_top_k: int) -> None:
    for model_kind, title in [
        ("lgbm_multiclass", "LightGBM multiclass"),
        ("xgb_multiclass", "XGBoost multiclass"),
        ("logistic", "Logistic regression benchmark"),
    ]:
        experiments.append(
            Experiment(
                exp_id=f"S3_{model_kind.upper()}",
                stage="stage3_models",
                title=title,
                target="target_excess_xfn_5d",
                feature_mode=best_feature_mode,
                model_kind=model_kind,
                threshold=best_threshold,
                adaptive_top_k=adaptive_top_k,
            )
        )


def run_experiments() -> Dict[str, object]:
    df = load_dataset()
    selection_cache: Dict[Tuple[float, str], Dict[str, object]] = {}
    feature_manifest: Dict[str, Dict[str, object]] = {}
    confusion_matrix_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    fixed_core_features: List[str] = []
    stable_feature_ranking: List[str] = []
    stable_core_stats_df = pd.DataFrame()

    experiments = build_round2_plan()
    pending_stage2 = True
    pending_stage3 = True

    index = 0
    while index < len(experiments):
        exp = experiments[index]
        index += 1

        for fold_id, train_end, test_start, test_end in E10_FOLDS:
            train_mask = (df["date"] <= pd.Timestamp(train_end)) & df[exp.target].notna()
            test_mask = (
                (df["date"] >= pd.Timestamp(test_start))
                & (df["date"] <= pd.Timestamp(test_end))
                & df[exp.target].notna()
            )

            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()
            features = resolve_features(
                exp=exp,
                df=df,
                train_df=train_df,
                fold_id=fold_id,
                fixed_core_features=fixed_core_features,
                stable_feature_ranking=stable_feature_ranking,
                selection_cache=selection_cache,
                feature_manifest=feature_manifest,
            )

            x_train, x_test, transform_meta = prepare_model_inputs(
                exp=exp,
                train_df=train_df,
                test_df=test_df,
                features=features,
            )
            if transform_meta:
                feature_manifest.setdefault(exp.exp_id, {}).setdefault(fold_id, {}).update(transform_meta)
            y_train_cont = train_df[exp.target].values
            y_test_cont = test_df[exp.target].values
            y_train_cls = label_3class(y_train_cont, threshold=exp.threshold)
            y_test_cls = label_3class(y_test_cont, threshold=exp.threshold)

            y_pred_cls = fit_predict(exp, x_train, y_train_cls, x_test)

            model_metrics = offset_metrics(y_test_cont, y_test_cls, y_pred_cls)
            majority_pred = baseline_majority(y_train_cls, len(y_test_cls))
            momentum_pred = label_3class(
                x_test["td_vs_xfn_5d"].values if "xfn" in exp.target else x_test["td_vs_tsx_5d"].values,
                threshold=exp.threshold,
            )
            majority_metrics = offset_metrics(y_test_cont, y_test_cls, majority_pred)
            momentum_metrics = offset_metrics(y_test_cont, y_test_cls, momentum_pred)
            calibration = full_row_calibration(y_test_cls, y_pred_cls)
            confusion_matrix_rows.extend(confusion_rows(exp.exp_id, fold_id, y_test_cls, y_pred_cls))

            fold_rows.append(
                {
                    "stage": exp.stage,
                    "exp_id": exp.exp_id,
                    "title": exp.title,
                    "target": exp.target,
                    "feature_mode": exp.feature_mode,
                    "model_kind": exp.model_kind,
                    "threshold": exp.threshold,
                    "feature_count": x_train.shape[1],
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

        current_summary = summarise_experiments(pd.DataFrame(fold_rows))

        if pending_stage2 and exp.stage == "stage1_band" and not (current_summary["stage"] == "stage1_band").any():
            continue
        if pending_stage2 and all(item.stage != "stage1_band" for item in experiments[index:]):
            best_band = choose_stage_winner(current_summary, "stage1_band")
            fixed_core_features, stable_core_stats_df = derive_stable_core_features(
                df=df,
                target="target_excess_xfn_5d",
                threshold=float(best_band["threshold"]),
                core_size=FIXED_CORE_SIZE,
            )
            stable_feature_ranking = stable_core_stats_df["feature"].tolist()
            append_feature_stage(experiments, float(best_band["threshold"]))
            pending_stage2 = False

        if pending_stage3 and not pending_stage2 and all(item.stage != "stage2_features" for item in experiments[index:]):
            best_features = choose_stage_winner(current_summary, "stage2_features")
            append_model_stage(
                experiments,
                best_threshold=float(best_features["threshold"]),
                best_feature_mode=str(best_features["feature_mode"]),
                adaptive_top_k=int(best_features.get("adaptive_top_k", 0) or 0),
            )
            pending_stage3 = False

    folds_df = pd.DataFrame(fold_rows)
    summary_df = summarise_experiments(folds_df)
    schema_df = build_feature_schema(df, fixed_core_features, stable_core_stats_df)
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
    confusion_df = pd.DataFrame(confusion_matrix_rows)

    band_winner = choose_stage_winner(summary_df, "stage1_band")
    feature_winner = choose_stage_winner(summary_df, "stage2_features")
    model_winner = choose_stage_winner(summary_df, "stage3_models")

    stage_selection = {
        "band_stage_winner": band_winner.to_dict(),
        "feature_stage_winner": feature_winner.to_dict(),
        "model_stage_winner": model_winner.to_dict(),
        "stable_core_features": fixed_core_features,
    }

    r07_baseline = summary_df.loc[summary_df["exp_id"] == "S1_B30"].iloc[0]
    stage_selection["beats_r07"] = bool(
        (model_winner["mean_active_sign_acc"] > r07_baseline["mean_active_sign_acc"])
        and (model_winner["mean_macro_f1"] >= r07_baseline["mean_macro_f1"] - 0.03)
        and (model_winner["mean_predicted_max_share"] <= 0.70)
    )

    return {
        "schema_df": schema_df,
        "stable_core_stats_df": stable_core_stats_df,
        "summary_df": summary_df,
        "folds_df": folds_df,
        "class_balance_df": class_balance_df,
        "confusion_df": confusion_df,
        "feature_manifest": feature_manifest,
        "stage_selection": stage_selection,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = run_experiments()

    schema_path = OUT_DIR / f"{ROUND2_PREFIX}_feature_schema.csv"
    schema_json_path = OUT_DIR / f"{ROUND2_PREFIX}_feature_schema.json"
    stable_core_stats_path = OUT_DIR / f"{ROUND2_PREFIX}_stable_core_stats.csv"
    summary_path = OUT_DIR / f"{ROUND2_PREFIX}_summary.csv"
    folds_path = OUT_DIR / f"{ROUND2_PREFIX}_fold_metrics.csv"
    features_path = OUT_DIR / f"{ROUND2_PREFIX}_feature_manifest.json"
    class_balance_path = OUT_DIR / f"{ROUND2_PREFIX}_class_balance.csv"
    confusion_path = OUT_DIR / f"{ROUND2_PREFIX}_confusion_matrices.csv"
    stage_selection_path = OUT_DIR / f"{ROUND2_PREFIX}_stage_selection.json"

    outputs["schema_df"].to_csv(schema_path, index=False)
    schema_json_path.write_text(outputs["schema_df"].to_json(orient="records", indent=2))
    outputs["stable_core_stats_df"].to_csv(stable_core_stats_path, index=False)
    outputs["summary_df"].to_csv(summary_path, index=False)
    outputs["folds_df"].to_csv(folds_path, index=False)
    outputs["class_balance_df"].to_csv(class_balance_path, index=False)
    outputs["confusion_df"].to_csv(confusion_path, index=False)
    features_path.write_text(json.dumps(outputs["feature_manifest"], indent=2))
    stage_selection_path.write_text(json.dumps(outputs["stage_selection"], indent=2))

    print("\nRound-2 single-stock redesign summary:\n")
    display_cols = [
        "stage",
        "exp_id",
        "feature_mode",
        "model_kind",
        "threshold",
        "mean_accuracy",
        "mean_macro_f1",
        "mean_active_sign_acc",
        "mean_active_coverage",
        "mean_majority_accuracy",
        "mean_momentum_accuracy",
        "uplift_vs_majority_pp",
        "uplift_vs_momentum_pp",
        "acceptable_candidate",
    ]
    print(outputs["summary_df"][display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved -> {schema_path}")
    print(f"Saved -> {schema_json_path}")
    print(f"Saved -> {stable_core_stats_path}")
    print(f"Saved -> {summary_path}")
    print(f"Saved -> {folds_path}")
    print(f"Saved -> {class_balance_path}")
    print(f"Saved -> {confusion_path}")
    print(f"Saved -> {features_path}")
    print(f"Saved -> {stage_selection_path}")


if __name__ == "__main__":
    main()
