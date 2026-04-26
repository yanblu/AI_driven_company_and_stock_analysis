# Event-Based Retraining Strategy
**TD Bank 5-Day Forward Return Model — Production Operations Plan**

---

## 1. Problem Statement

The current model uses calendar-based retraining (every 6 months, expanding window). This works well during stable regimes but has a structural blind spot: it cannot respond to mid-cycle regime shifts. Three observed failures illustrate this:

| Fold | Period | DirAcc | Root Cause | Calendar retrain helped? |
|---|---|---|---|---|
| v2 | 23H2 | 45.8% | AML settlement resolution — company-specific event | No — retrain was months away |
| v5 | 25H1 | 50.0% | Ambiguous transition post-recovery | Partially |
| v7 | 26Q1 | 46.2% | Tariff shock — first-of-kind macro event | No — model had never seen this regime |

**Core insight:** Calendar retraining assumes regimes shift gradually and predictably. In reality, regime shifts are triggered by discrete events. Event-based retraining aligns model updates with the actual information structure of markets.

---

## 2. Three-Trigger Framework

We propose three complementary trigger types that together cover the known failure modes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIGGER HIERARCHY                                │
│                                                                     │
│  Tier 1 — Proactive (known in advance)                              │
│  └── Earnings call trigger                                          │
│      Fires on a known date before the event                         │
│                                                                     │
│  Tier 2 — Reactive (detected from data)                             │
│  └── Macro fear trigger (VIX threshold)                             │
│      Fires when market conditions cross a threshold                  │
│                                                                     │
│  Tier 3 — Fallback (model self-monitoring)                          │
│  └── Drift trigger (rolling DirAcc monitor)                         │
│      Fires when model performance degrades regardless of cause      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Trigger Definitions

### Trigger 1 — Earnings Call (Proactive)

**What it solves:** v2-type failures. Earnings calls are the primary source of new NLP signals. After each quarterly call, the model's NLP features shift materially — forward guidance, sentiment tone, topic distribution. The current model operates for up to 6 months on stale post-call features before retraining.

**Fire condition:**
```
Date = TD quarterly earnings call date (known ~3 weeks in advance)
Action fires: on the evening of the earnings call date (after market close)
```

**Earnings call schedule (historical, for reference):**
- Q1: typically last week of February
- Q2: typically last week of May
- Q3: typically last week of August
- Q4: typically first week of December

**Retrain scope:** Full expanding window through earnings call date. NLP features are re-activated with new call data.

**Signal suppression:** 5 trading days post-retrain (one target horizon). The model needs at least one full prediction cycle before being trusted.

---

### Trigger 2 — Macro Fear Threshold (Reactive)

**What it solves:** v7-type failures. A sudden macro shock (tariff announcement, geopolitical event, central bank surprise) shifts the feature-to-return relationship before any calendar retrain would fire.

**Fire condition:**
```
VIX 5-day moving average crosses 28 upward  
  AND  
DXY 5-day return > +1.5% (dollar surge, risk-off confirmed)  
  AND  
The trigger has not fired in the past 30 trading days (cooldown to avoid re-firing)
```

**Why both conditions:** VIX alone spikes briefly all the time (earnings, data releases). Requiring concurrent DXY confirmation filters for genuine broad-based risk-off events, not just single-day volatility.

**Retrain scope:** Expanding window through trigger date. The model immediately incorporates the new fear regime.

**Signal suppression:** 15 trading days post-retrain. Macro shocks take longer to stabilize than earnings — the model needs more runway to validate before full confidence is restored.

**Confidence ramp-up:**
```
Days 1–5:   No signals emitted (full suppression)
Days 6–10:  Emit signals at 50% position size
Days 11–15: Emit signals at 75% position size
Day 16+:    Full signals, subject to drift monitor
```

---

### Trigger 3 — Drift Monitor (Fallback)

**What it solves:** Anything the above two triggers miss — gradual drift, company-specific events not tied to earnings, slow regime shifts. This is the safety net that catches everything.

**Fire condition:**
```
Rolling 20-prediction DirAcc (stride-5, independent observations) < 48%
  AND
The trigger has not fired in the past 20 trading days (cooldown)
```

**Why 48% and not 50%:** Sampling variance on 20 observations means a true 50% model will occasionally read 48% by chance. Using 48% reduces false positives while still catching genuine degradation. Over 20 independent observations, 48% means approximately 9–10 correct out of 20 — meaningfully below random.

**Retrain scope:** Full expanding window through current date.

**Signal suppression:** 10 trading days. After drift-triggered retraining, suppress signals while we observe if the retrain stabilized performance.

**Escalation:** If DirAcc remains < 48% after the retrain and suppression window, emit a manual review alert. Do not retrain again automatically — this indicates a structural regime shift that requires human judgment.

---

## 4. Interaction Between Triggers

The three triggers can fire independently. Rules for concurrent firings:

| Situation | Action |
|---|---|
| Earnings + Macro fire within 5 days of each other | Use the later date as the retrain date (incorporates more data) |
| Drift fires while in a suppression window | Extend suppression window, do not retrain again |
| Earnings fires during Macro suppression window | Retrain (earnings is additive new information), reset suppression clock |
| All three fire simultaneously | Retrain once on current date, apply the longest suppression window (15 days) |

---

## 5. Implementation Phases

### Phase 0 — Prerequisites (before any trigger logic)
- [ ] Refactor `src/models/baseline.py` so retraining can be invoked as a standalone function with a `cutoff_date` parameter
- [ ] Add a `prediction_log.parquet` that stores every daily prediction with timestamp (needed by drift monitor)
- [ ] Add a `retrain_log.csv` that records trigger type, fire date, retrain date, and post-retrain performance

### Phase 1 — Earnings Trigger (lowest effort, highest ROI)
- [ ] Extract all historical earnings call dates from `data/processed/llm_annotations/transcripts.jsonl` (already available)
- [ ] Build `src/ops/trigger_earnings.py`: checks if today is an earnings call date → calls retrain function
- [ ] Add 5-day suppression flag to daily signal output
- [ ] Test: simulate v2 fold by triggering retrain on AML-period earnings call dates and measuring DirAcc change

### Phase 2 — Drift Monitor (safety net, medium effort)
- [ ] Build `src/ops/trigger_drift.py`: reads last 20 entries from `prediction_log.parquet`, computes rolling DirAcc, fires if < 48%
- [ ] Add alert email/log output when drift fires
- [ ] Add escalation path if post-retrain DirAcc still < 48%
- [ ] Test: simulate v7 fold to confirm drift fires within the first 10 degraded trading days

### Phase 3 — Macro Fear Trigger (medium effort, addresses black-swan events)
- [ ] Build `src/ops/trigger_macro.py`: pulls daily VIX and DXY from existing macro data, checks threshold conditions
- [ ] Implement 30-day cooldown and confidence ramp-up
- [ ] Backtest: confirm trigger would have fired on or around the tariff shock period in 26Q1
- [ ] Tune VIX threshold (28 is initial estimate — adjust based on false positive rate in backtest)

### Phase 4 — Orchestration (production hardening)
- [ ] Build `src/ops/daily_run.py`: master script that checks all three triggers, executes retrain if needed, logs everything
- [ ] Add `prediction_log.parquet` append-on-prediction logic
- [ ] Unit tests for each trigger condition
- [ ] Documentation for operations team (how to interpret the retrain log, how to manually override)

---

## 6. Expected Impact on Failing Folds

| Fold | Failure cause | Which trigger | Expected improvement |
|---|---|---|---|
| v2 (23H2) | Post-AML recovery | Earnings trigger | Model retrained on Q2/Q3 2023 earnings → incorporates recovery-period NLP signals. Estimated +4–6pp |
| v5 (25H1) | Ambiguous transition | Earnings trigger | Mild benefit from quarterly refresh. v5 is at 50% (honest uncertainty), may not improve much |
| v7 (26Q1) | Tariff shock | Macro + Drift triggers | Drift fires within ~10 days of shock → suppression prevents worst losses. Post-suppression model still limited by training data scarcity. Estimated -3pp losses prevented, not +pp gains |

**Key expectation setting:** Event-based retraining prevents *continued* losses in bad regimes by suppressing signals. It does not magically improve predictions in genuinely novel regimes (v7). The v7 tariff shock improvement is loss prevention, not accuracy improvement.

---

## 7. Limitations and Known Risks

1. **Lookahead risk on trigger design:** The VIX/DXY thresholds (28 / +1.5%) were chosen partly by observing v7. In production, these must be fixed before deployment and not re-tuned on observed failures — otherwise we're fitting to the test set.

2. **Suppression opportunity cost:** Every day in suppression is a day with no signal. If the macro trigger fires during a period that turns out to be a false alarm (VIX spike that recovers in 3 days), we lose 15 days of valid signal. The confidence ramp-up mitigates this but doesn't eliminate it.

3. **Data availability lag:** Earnings call transcripts may not be fully processed the evening of the call. Allow 1 business day lag between call date and retrain execution to ensure NLP features are complete.

4. **Minimum training data requirement:** Expanding window retraining works as long as the training set has > 200 rows. The first trigger that would fire (Q1 2021 earnings) would have ~60 rows — too few for LGBM. Do not allow event-triggered retraining until the training set has at least 200 rows (approximately end of 2021).

---

## 8. Success Metrics

The event-based retraining strategy is considered successful if, in backtest simulation:

| Metric | Target |
|---|---|
| Mean DirAcc across 7 folds | > 57% (vs. 55.7% baseline) |
| Weak-fold DirAcc (v2, v7) | > 50% (vs. 45.8%, 46.2% baseline) |
| Days in suppression | < 15% of total trading days |
| False trigger rate (Macro trigger fires but DirAcc was fine) | < 2 per year |

---

## 9. Relationship to Calendar Retraining

Event-based triggers **supplement** the existing 6-month calendar retrain, they do not replace it. The calendar retrain ensures the model incorporates gradual drift over time even when no discrete events fire. The event triggers ensure the model responds rapidly to discrete shocks.

```
Timeline example:
  Feb 2026     May 2026     Aug 2026     Nov 2026
    │            │            │            │
    ▼            ▼            ▼            ▼
  [Q1 earnings] [Q2 earnings] [Q3 earnings] [Q4 earnings]  ← Tier 1: always fires
       │                           │
       ▼                           ▼
  [Calendar retrain]          [Calendar retrain]           ← Calendar: every 6 months
  
  If VIX/DXY threshold crossed at any point → Tier 2 fires immediately
  If rolling DirAcc < 48% at any point → Tier 3 fires immediately
```
