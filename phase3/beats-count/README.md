# beats-count — does X1f need all three BEATs networks?

> **Post hoc (phase-3 rule 2).** No network is trained. Every member's eval scores have been seen,
> and the closest earlier result is also seen: adding one BEATs seed to E2′[A8] gave eval
> +0.10 / +0.18 / +0.24, against +0.37 for all three (`ablation/beats`). The systems below have
> **not** been scored: unlike `../beats-router`, swapping BEATs members changes every machine's
> scores, so the point estimates are not arithmetic and the predictions are real.

## Question

X1f = GS12f (12 from-scratch networks) + 3 BEATs networks. The BEATs block is 83 % of X1f's forward
FLOPs (`../beats-router/scores/cost.json`). Three BEATs networks were never chosen as the best
number. Three was the budget of `ablation/beats`: 3 GPUs, 3 seeds, and a "≥ 2 of 3" decision rule.
Does X1f keep its eval accuracy with one or two BEATs networks?

Cost is arithmetic, from `../beats-router/scores/cost.json` (forward only, fraction of X1f):

| BEATs nets k | FLOPs | latency, batch 32 |
|---|---|---|
| 1 | **0.446** | 0.509 |
| 2 | 0.723 | 0.755 |
| 3 (X1f) | 1 | 1 |

## Setup

- **From-scratch block**: GS12f exactly, with 8 `fc_nogain` 2060–67 and 4 `fc_spec` 2060–63,
  S1′ each, weight 1. BEATs members are `bt_arc_s2040–42`, zK′ each. The machinery is
  `synthesis/ensemble-g8/score.py`, imported; no new scoring rule. Dumps come from `../OS-SCL-syn-fresh`
  via a symlink tree, as in `../beats-router`.
- **Subsets**: every subset of the 3 BEATs nets of size k. That is three subsets for k = 1 (B0, B1, B2)
  and three for k = 2 (B01, B02, B12). k = 3 is X1f itself.
- **Weighting**:
  - **primary, "block"**: the BEATs block keeps X1f's total weight of 4.5 (r = 0.375 of the
    from-scratch weight 12), so the per-net weight is 4.5 / k. The ratio r was picked on dev for the
    block as a whole (`synthesis/grand-ensemble`).
  - **secondary, "member"**: per-net weight stays 1.5, so the block's share shrinks with k.
- **CIs**: `grand-ensemble`'s paired clip bootstrap, with the same draws and seed.

### Selection (dev only, two stages)

1. `score.py dev` computes every subset under both weightings **on dev only**.
2. For each k ∈ {1, 2}, the deployed system **X1f-k** is the subset with the highest dev Amean
   under the primary weighting. A tie within 0.05 goes to the subset with the lower seed numbers.
3. The picks are written to `scores/selection.json` and **committed** before `score.py eval` runs,
   and `score.py eval` refuses to run otherwise. Eval is scored once, replicated byte for byte.

### Decision rules

For each k ∈ {1, 2}, **D1_k** passes if all three hold:

- (a) the eval paired CI of X1f-k − X1f has its lower bound ≥ −0.20 Amean. This is `../beats-router`'s
  margin: one step above the ~0.15 ensemble noise floor;
- (b) at least 2 of the 3 size-k subsets have eval Δ vs X1f ≥ −0.10 (primary weighting), so the
  result does not hinge on which seed was kept;
- (c) the eval paired CI of X1f-k − GS12f excludes 0 from above.

**Recommendation.** The smallest k that passes D1_k is named "X1f-lite", the low-cost variant reported
beside X1f. If neither passes, X1f stays as it is. X1f stays the reported accuracy number either way.

**D2, descriptive.** For each k, mean eval Amean over the subsets, against k, under both weightings.
It is called "diminishing returns" if mean(k=2) − mean(k=1) > X1f − mean(k=2).

## Predictions (written before anything is computed; 80 % intervals)

| ID | quantity | prediction |
|---|---|---|
| Q1 | mean over k = 1 subsets, eval Amean, primary | 95.30 [95.15, 95.45] |
| Q2 | mean over k = 2 subsets, eval Amean, primary | 95.44 [95.33, 95.53] |
| Q3 | X1f-1 − X1f eval CI lower bound | −0.32 [−0.45, −0.18] → D1_1 **fails** (70 %) |
| Q4 | X1f-2 − X1f eval CI lower bound | −0.17 [−0.25, −0.08] → D1_2 **passes** (55 %) |
| Q5 | D2 diminishing returns (primary) | yes (70 %) |
| Q6 | secondary minus primary weighting, k = 1 mean eval | −0.05 [−0.20, +0.10] |
| Q7 | the dev pick for k = 1 is also the best k = 1 subset on eval | 40 % (chance is 33 %) |

## Results

*(not yet computed)*
