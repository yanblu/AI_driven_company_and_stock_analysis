# Step 2 — LLM Company Analysis

AI-driven company analysis using earnings call transcripts, news, and filings for TD Bank.

## Contents

| File / Folder | Description |
|---|---|
| `02_annotation.ipynb` | Main notebook: LLM annotation pipeline, sentiment/topic extraction, FinBERT vs LLM comparison |
| `step2_analysis.md` | Full written analysis: findings on TD's tone, topics, guidance, AML signals, analyst sentiment |
| `canvases/finbert-vs-llm.canvas.tsx` | Interactive comparison: FinBERT vs GPT-4 sentiment scoring |
| `canvases/holistic-analysis.canvas.tsx` | Holistic company dashboard |
| `src_analysis/` | Python source for annotation, aggregation, and FinBERT baseline |

## Key Findings (from `step2_analysis.md`)
1. **CEO/CFO tone divergence** — CFO tone leads CEO tone in predicting analyst sentiment changes, suggesting financial-framing matters more than management framing post-2022.
2. **AML regulatory pressure** — AML topic share spiked significantly in 2023-2024, coinciding with underperformance vs sector peers.
3. **Guidance reliability** — Q&A-session tone systematically lower than prepared remarks; the gap (`evt_framing_gap`) is a persistent signal.
4. **Source dispersion** — Agreement/disagreement across CEO, CFO, and news sentiment is informative about information environment uncertainty.

## LLM Features Produced
These features feed into the Step 3 predictive model:

| Feature | Construction |
|---|---|
| `evt_ceo_tone` | CEO prepared-statement LLM sentiment, decayed from call date |
| `evt_cfo_tone` | CFO prepared-statement LLM sentiment, decayed |
| `evt_framing_gap` | CEO prepared minus CEO Q&A sentiment |
| `evt_guidance_strength` | Forward-guidance sentiment strength |
| `evt_aml_pressure` | AML regulatory topic share |
| `evt_topic_entropy` | Shannon entropy of topic distribution |
| ... | (11 total, see model card) |
