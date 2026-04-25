"""
Data quality check for model_features_daily.parquet.
Prints a structured report to stdout. Review before running models.
"""
from pathlib import Path
import numpy as np
import pandas as pd

STEP1_DIR = Path(__file__).resolve().parents[2]
PATH = STEP1_DIR / "data/features/model_features_daily.parquet"

CALL_DATES = {
    "FY2021Q1": "2021-02-25", "FY2021Q2": "2021-05-27", "FY2021Q3": "2021-08-26",
    "FY2021Q4": "2021-12-02", "FY2022Q1": "2022-03-03", "FY2022Q2": "2022-05-26",
    "FY2022Q3": "2022-08-25", "FY2022Q4": "2022-12-01", "FY2023Q1": "2023-03-02",
    "FY2023Q2": "2023-05-26", "FY2023Q3": "2023-08-24", "FY2023Q4": "2023-11-30",
    "FY2024Q1": "2024-02-29", "FY2024Q2": "2024-05-23", "FY2024Q3": "2024-08-22",
    "FY2024Q4": "2024-12-05", "FY2025Q1": "2025-02-27", "FY2025Q2": "2025-05-22",
    "FY2025Q3": "2025-08-28", "FY2025Q4": "2025-12-04", "FY2026Q1": "2026-04-03",
}

TARGET = "td_return_5d_fwd"
CUTOFF_A = pd.Timestamp("2024-12-31")
CUTOFF_C = pd.Timestamp("2024-08-22")

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def subsection(title: str):
    print(f"\n--- {title} ---")

df = pd.read_parquet(PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
feature_cols = [c for c in df.columns if c != TARGET and c != "date"]

# ── 1. Overview ──────────────────────────────────────────────────────────────
section("1. OVERVIEW")
print(f"  Rows:             {len(df):,}")
print(f"  Columns total:    {df.shape[1]}  (1 date + {len(feature_cols)} features + 1 target)")
print(f"  Date range:       {df['date'].min().date()} → {df['date'].max().date()}")

train_a = df[df["date"] <= CUTOFF_A]
test_a  = df[df["date"] >  CUTOFF_A]
train_c = df[df["date"] <= CUTOFF_C]
test_c  = df[df["date"] >  CUTOFF_C]
print(f"\n  Option A split (cutoff {CUTOFF_A.date()}):")
print(f"    Train: {len(train_a):,} rows  ({train_a['date'].min().date()} → {train_a['date'].max().date()})")
print(f"    Test:  {len(test_a):,} rows  ({test_a['date'].min().date()} → {test_a['date'].max().date()})")
print(f"\n  Option C split (cutoff {CUTOFF_C.date()}):")
print(f"    Train: {len(train_c):,} rows  ({train_c['date'].min().date()} → {train_c['date'].max().date()})")
print(f"    Test:  {len(test_c):,} rows  ({test_c['date'].min().date()} → {test_c['date'].max().date()})")

# ── 2. Target distribution ───────────────────────────────────────────────────
section("2. TARGET — td_return_5d_fwd")
tgt = df[TARGET]
print(f"  Mean:    {tgt.mean():.4f}  ({tgt.mean()*100:.2f}%)")
print(f"  Std:     {tgt.std():.4f}  ({tgt.std()*100:.2f}%)")
print(f"  Min:     {tgt.min():.4f}  ({tgt.min()*100:.2f}%)")
print(f"  Max:     {tgt.max():.4f}  ({tgt.max()*100:.2f}%)")
print(f"  % positive: {(tgt > 0).mean()*100:.1f}%")
print(f"  % negative: {(tgt < 0).mean()*100:.1f}%")

subsection("Target around key events")
events = {
    "AML consent order (Oct 10 2024)": "2024-10-10",
    "FY2024Q4 call (Dec 5 2024)":       "2024-12-05",
    "FY2025Q1 call (Feb 27 2025)":       "2025-02-27",
}
for label, dt in events.items():
    row = df[df["date"] == pd.Timestamp(dt)]
    if len(row):
        val = row[TARGET].iloc[0]
        print(f"  {label}: {val*100:+.2f}%")
    else:
        print(f"  {label}: date not in dataset (non-trading day)")

# ── 3. Missing values ────────────────────────────────────────────────────────
section("3. MISSING VALUES")
missing = df[feature_cols].isnull().sum()
total_cells = len(df) * len(feature_cols)
print(f"  Total missing cells: {missing.sum():,} / {total_cells:,}  ({missing.sum()/total_cells*100:.2f}%)")
print(f"  Columns with any missing: {(missing > 0).sum()} / {len(feature_cols)}")

subsection("All columns with missing values")
miss_pct = (missing[missing > 0] / len(df) * 100).sort_values(ascending=False)
for col, pct in miss_pct.items():
    print(f"  {col:<60s}  {missing[col]:>5d}  ({pct:.1f}%)")

# ── 4. Feature groups ────────────────────────────────────────────────────────
section("4. FEATURE GROUPS")
price_cols  = [c for c in feature_cols if c.startswith("td_") or c.startswith("xfn_") or c.startswith("tsx_")]
fx_cols     = [c for c in feature_cols if c.startswith("fx_")]
rate_cols   = [c for c in feature_cols if c.startswith("rate_") or c.startswith("yield_")]
cpi_cols    = [c for c in feature_cols if c.startswith("cpi_")]
news_cols   = [c for c in feature_cols if c.startswith("news_") or c.startswith("days_since_last")]
event_cols  = [c for c in feature_cols if c.startswith("is_")]
nlp_cols    = [c for c in feature_cols if c not in set(
    price_cols + fx_cols + rate_cols + cpi_cols + news_cols + event_cols
)]

groups = [
    ("Price / momentum (TD, XFN, GSPTSE)", price_cols),
    ("FX (USD/CAD)",                        fx_cols),
    ("Rates / yields",                      rate_cols),
    ("CPI",                                 cpi_cols),
    ("News rolling",                        news_cols),
    ("Event flags",                         event_cols),
    ("NLP engineered",                      nlp_cols),
]
for name, cols in groups:
    print(f"  {name:<45s}  {len(cols):>4d} columns")

# ── 5. NLP point-in-time spot-check ─────────────────────────────────────────
section("5. NLP POINT-IN-TIME SPOT-CHECK")
print("  Verifying CEO prep sentiment forward-fill is activated on call date,")
print("  not before, and updates correctly at each subsequent call.\n")

ceo_col = "transcript_ceo_prep_sentiment_mean_ffill"
call_list = sorted(CALL_DATES.items(), key=lambda x: x[1])

print(f"  {'Quarter':<12} {'Call date':<13} {'Day before call':<18} {'Call date val':<15} {'Day after call'}")
print(f"  {'-'*12} {'-'*13} {'-'*18} {'-'*15} {'-'*15}")
for fq, dt in call_list:
    call_ts = pd.Timestamp(dt)
    day_before = df[df["date"] < call_ts].tail(1)
    day_of     = df[df["date"] == call_ts]
    day_after  = df[df["date"] > call_ts].head(1)
    if ceo_col not in df.columns:
        print(f"  Column '{ceo_col}' not found.")
        break
    b = f"{day_before[ceo_col].iloc[0]:.3f}" if len(day_before) and pd.notna(day_before[ceo_col].iloc[0]) else "n/a"
    o = f"{day_of[ceo_col].iloc[0]:.3f}"     if len(day_of)     and pd.notna(day_of[ceo_col].iloc[0])     else "n/a"
    a = f"{day_after[ceo_col].iloc[0]:.3f}"  if len(day_after)  and pd.notna(day_after[ceo_col].iloc[0])  else "n/a"
    print(f"  {fq:<12} {dt:<13} {b:<18} {o:<15} {a}")

# ── 6. Key NLP feature stats per quarter window ──────────────────────────────
section("6. KEY NLP FEATURES — QUARTERLY WINDOW STATS")
print("  For each inter-call window: mean CEO prep sentiment, AML share, guidance share")
print(f"\n  {'Quarter':<12} {'CEO prep (ffill)':<18} {'AML share (ffill)':<20} {'Guidance share (ffill)'}")
print(f"  {'-'*12} {'-'*18} {'-'*20} {'-'*22}")

call_tss = sorted([pd.Timestamp(v) for v in CALL_DATES.values()])
for i, start in enumerate(call_tss):
    end = call_tss[i+1] - pd.Timedelta(days=1) if i+1 < len(call_tss) else df["date"].max()
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(window) == 0:
        continue
    fq = [k for k, v in CALL_DATES.items() if pd.Timestamp(v) == start]
    fq_label = fq[0] if fq else "?"
    ceo  = window.get("transcript_ceo_prep_sentiment_mean_ffill", pd.Series()).mean()
    aml  = window.get("topic_regulatory_AML_share_ffill", pd.Series()).mean()
    guid = window.get("topic_guidance_share_ffill", pd.Series()).mean()
    print(f"  {fq_label:<12} {ceo:<18.3f} {aml:<20.3f} {guid:.3f}")

# ── 7. News feature sanity check ─────────────────────────────────────────────
section("7. NEWS FEATURES — SANITY CHECK")
if "news_sent_mean_7d" in df.columns:
    news_s = df["news_sent_mean_7d"].dropna()
    print(f"  news_sent_mean_7d:   mean={news_s.mean():.3f}  std={news_s.std():.3f}  "
          f"min={news_s.min():.3f}  max={news_s.max():.3f}  "
          f"coverage={len(news_s)/len(df)*100:.1f}%")
if "news_sent_mean_30d" in df.columns:
    news_30 = df["news_sent_mean_30d"].dropna()
    print(f"  news_sent_mean_30d:  mean={news_30.mean():.3f}  std={news_30.std():.3f}  "
          f"min={news_30.min():.3f}  max={news_30.max():.3f}  "
          f"coverage={len(news_30)/len(df)*100:.1f}%")
if "news_count_7d" in df.columns:
    print(f"  news_count_7d:       mean={df['news_count_7d'].mean():.1f}  "
          f"max={df['news_count_7d'].max():.0f}  "
          f"days with burst (>10): {(df['news_count_7d']>10).sum()}")

subsection("Dates with highest 7-day news volume (top 5)")
if "news_count_7d" in df.columns:
    top_news = df.nlargest(5, "news_count_7d")[["date", "news_count_7d", "news_sent_mean_7d", TARGET]]
    print(top_news.to_string(index=False))

# ── 8. Duplicate / constant columns check ────────────────────────────────────
section("8. CONSTANT OR NEAR-CONSTANT COLUMNS")
std_vals = df[feature_cols].std()
constant = std_vals[std_vals == 0].index.tolist()
near_const = std_vals[(std_vals > 0) & (std_vals < 1e-6)].index.tolist()
print(f"  Fully constant columns:      {len(constant)}")
if constant:
    print(f"    {constant}")
print(f"  Near-constant columns (σ<1e-6): {len(near_const)}")
if near_const:
    print(f"    {near_const}")

# ── 9. Sample rows around key dates ──────────────────────────────────────────
section("9. SAMPLE ROWS — KEY DATES")
key_dates = ["2024-10-10", "2024-12-05", "2025-02-27"]
show_cols = ["date", TARGET, "td_return_5d", "td_vs_xfn_5d",
             "transcript_ceo_prep_sentiment_mean_ffill",
             "topic_regulatory_AML_share_ffill",
             "topic_guidance_share_ffill",
             "news_sent_mean_7d", "days_since_call"]
show_cols = [c for c in show_cols if c in df.columns]
for dt in key_dates:
    rows = df[df["date"] == pd.Timestamp(dt)]
    if len(rows):
        print(f"\n  {dt}:")
        for col in show_cols:
            val = rows[col].iloc[0]
            print(f"    {col:<50s} {val}")
    else:
        print(f"\n  {dt}: not a trading day")

print("\n" + "="*65)
print("  Data quality check complete.")
print("="*65)
