# Brand Selection Report — Resolv (Milestone 2)

All numbers below come from `evaluation/brand_profile.csv`, produced by
`scripts/profile_brands.py`, which reuses the Milestone 1 cached artifacts
(`data/raw/twcs_cache.pkl`, `data/raw/twcs_components.pkl`) — no raw CSV
reload. Reproduce with:

```
python scripts/profile_brands.py
```

## 1. Comparison table (measured)

| brand | components | customer tweets | brand replies | % w/ brand response | median len | mean len | % multi-turn (>2) | usable cases | distinct customer msgs | dup rate % |
|---|---|---|---|---|---|---|---|---|---|---|
| AmazonHelp | 82,556 | 203,598 | 169,840 | 100.00 | 3.0 | 4.531 | 62.09 | 82,556 | 196,371 | 3.55 |
| AppleSupport | 80,717 | 131,764 | 106,860 | 100.00 | 2.0 | 2.960 | 34.87 | 80,717 | 126,351 | 4.11 |
| Uber_Support | 41,923 | 72,154 | 56,270 | 100.00 | 2.0 | 3.066 | 35.99 | 41,923 | 69,348 | 3.89 |
| SpotifyCares | 28,280 | 48,543 | 43,265 | 99.99 | 2.0 | 3.249 | 37.13 | 28,277 | 46,649 | 3.90 |
| AmericanAir | 26,386 | 50,054 | 36,764 | 100.00 | 2.0 | 3.319 | 43.88 | 26,386 | 49,062 | 1.98 |
| Delta | 26,168 | 45,296 | 42,253 | 99.99 | 2.0 | 3.363 | 42.60 | 26,166 | 43,976 | 2.91 |
| comcastcares | 24,063 | 39,579 | 33,031 | 100.00 | 2.0 | 3.036 | 36.55 | 24,063 | 38,115 | 3.70 |
| TMobileHelp | 22,820 | 47,158 | 34,317 | 100.00 | 2.0 | 3.625 | 37.80 | 22,820 | 45,325 | 3.89 |
| SouthwestAir | 21,636 | 35,370 | 28,977 | 99.99 | 2.0 | 2.992 | 33.42 | 21,634 | 34,575 | 2.25 |
| Ask_Spectrum | 18,532 | 33,252 | 25,860 | 99.99 | 2.0 | 3.218 | 35.40 | 18,531 | 32,273 | 2.94 |

**"% w/ brand response"** = share of that brand's conversation components containing
≥1 customer tweet *and* ≥1 reply from that same brand (i.e., a usable Q→A pair).
It is ~100% for every candidate almost by construction (components were selected
*because* the brand replied in them), so it does not discriminate between brands —
included for completeness, not as a differentiator.

**Diversity clustering (TF-IDF + KMeans, k=8, capped at 3,000 sampled distinct
customer messages per brand) — labeled APPROX in the CSV:** silhouette scores
came back near zero for every single brand (0.014–0.023) and cluster-balance
scores clustered tightly around 0.40–0.53 with no brand standing out. **This
signal is not discriminative** — short, noisy tweet text doesn't separate
cleanly under TF-IDF/KMeans at this scale, so I'm not using it to claim any
brand has "more diverse" issues than another. It's reported for transparency,
not used as a selection criterion.

## 2. Measured statistics — notable patterns

- **AmazonHelp** has both the largest volume *and* the richest conversations:
  highest mean length (4.53 vs. 2.96–3.63 for others) and by far the highest
  multi-turn rate (62.09% vs. 33–44% elsewhere). It also has the most distinct
  customer messages (196,371) — the most raw material to derive an intent
  taxonomy from real data rather than guessing.
- **AppleSupport** is close in volume but structurally simpler: 65% of its
  conversations are a single Q→A exchange (34.87% multi-turn), and it has the
  highest duplicate rate of the ten (4.11%).
- The three airlines (AmericanAir, Delta, SouthwestAir) have the **lowest
  duplicate rates** (1.98–2.91%) — cleanest raw text — but roughly a third the
  volume of AmazonHelp/AppleSupport.
- All ten candidates clear "% w/ brand response" at ~100%, so response coverage
  is not a differentiator — every candidate has enough grounded historical
  replies to retrieve from.

## 3. Advantages / disadvantages of the strongest candidates

**AmazonHelp**
- \+ Largest historical-evidence pool for retrieval grounding (169,840 replies)
- \+ Highest multi-turn rate — better material for testing multi-turn context handling and escalation logic, which the assignment explicitly wants demonstrated
- \+ Largest, most distinct customer-message pool for deriving intents from data
- − Amazon support is known to span multiple sub-businesses (retail orders, devices, Prime Video, payments) — intent taxonomy could turn out broader/messier than a single-product brand; needs to be checked empirically in Milestone 3, not assumed
- − Moderate duplicate rate (3.55%), not the cleanest of the ten

**AppleSupport**
- \+ Second-largest volume, single clear product ecosystem (likely narrower, more coherent intents)
- − Structurally simpler conversations (65% single-turn) — less material to demonstrate escalation/multi-turn handling
- − Highest duplicate rate of the ten

**TMobileHelp**
- \+ Second-highest mean conversation length (3.625) after Amazon
- − Roughly a quarter of AmazonHelp's volume — thinner golden-set sampling pool for rare/ambiguous cases

## 4. Recommended brand

**AmazonHelp**

## 5. Why AmazonHelp for this assignment specifically

- **Golden-set feasibility:** 82,556 usable conversations and 196,371 distinct
  customer messages give the most headroom to sample a genuinely varied
  150–250 example golden set (common + rare + ambiguous cases) without running
  out of material, which is the actual bottleneck on smaller candidates like
  TMobileHelp or Ask_Spectrum.
- **Multi-turn structure demonstrates the escalation/context components:**
  62.09% of AmazonHelp conversations have more than 2 messages, the highest of
  any candidate — this is the only brand where "conversation context
  preprocessing" and multi-turn grounding are exercised often enough to
  produce meaningful failure cases, rather than being a thin layer over what's
  mostly single Q→A pairs (AppleSupport, SouthwestAir).
- **Retrieval grounding depth:** 169,840 historical brand replies is the
  largest evidence corpus of the ten, which matters directly for the
  evidence-retrieval and grounded-generation stages — more candidates to
  retrieve from and rerank means retrieval metrics (Recall@k, MRR) will be
  measuring something real rather than saturating on a thin corpus.

**Caveat:** the diversity-clustering metric could not confirm or deny that
Amazon's issues are topically more varied than other brands' — it was
inconclusive across all ten candidates and is not part of this reasoning. The
case for AmazonHelp rests on volume, distinct-message count, and multi-turn
structure, all directly measured. It's also possible AmazonHelp's breadth
(retail vs. devices vs. Prime Video, etc.) makes intent discovery in
Milestone 3 messier than a narrower single-product brand — that will be
checked empirically against the actual data, not assumed here.
