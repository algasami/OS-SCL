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

> **Verdict: one BEATs network is enough. D1_1 passes, so X1f-lite = X1f-1 = GS12f + BEATs s2040
> at block weight 4.5.**
> - It uses 0.446 of X1f's forward FLOPs (0.51 of batch-32 latency) and scores eval 95.44 against
>   X1f's 95.51: Δ −0.07 [−0.15, +0.01].
> - The CI includes 0, and the loss is under the ~0.15 seed-noise floor.
> - All three single-BEATs choices land within 0.09 of X1f on eval, so the result does not depend
>   on which seed is kept.
> - **X1f stays the reported accuracy number.** X1f-lite is its low-cost variant.

The dev picks were committed before eval was scored (7fc9609); eval was scored once, in 2c5147c.
Both stages were replicated byte for byte. X1f and GS12f reproduce `ensemble-g8` to ≤ 4.4e-5.
k = 3 under both weightings is bit-identical to X1f.

### Systems (Amean, dev / eval)

| BEATs nets | subset | block weight 4.5 (primary) | 1.5 per net (secondary) | FLOPs vs X1f |
|---|---|---|---|---|
| 0 | GS12f | 95.01 / 95.03 | — | 0.169 |
| 1 | **B0 = X1f-1** (dev pick) | 95.20 / **95.44** | 95.18 / 95.30 | 0.446 |
| 1 | B1 | 95.12 / 95.42 | 95.15 / 95.28 | 0.446 |
| 1 | B2 | 95.18 / 95.46 | 95.14 / 95.32 | 0.446 |
| 2 | **B01 = X1f-2** (dev pick, tie rule) | 95.19 / **95.48** | 95.19 / 95.42 | 0.723 |
| 2 | B02 | 95.20 / 95.51 | 95.20 / 95.46 | 0.723 |
| 2 | B12 | 95.17 / 95.48 | 95.18 / 95.43 | 0.723 |
| 3 | X1f | 95.19 / 95.51 | (same) | 1 |

All three k = 2 subsets were within 0.05 on dev, so the tie rule picked B01, the lowest seed numbers.
B02 was the highest on dev by 0.009.

### Decisions (eval, paired 95 % CI)

| rule | X1f-1 (B0) | X1f-2 (B01) |
|---|---|---|
| (a) CI of Δ vs X1f, lower bound ≥ −0.20 | −0.07 [**−0.155**, +0.009]: pass | −0.04 [**−0.095**, +0.030]: pass |
| (b) ≥ 2 of 3 subsets with Δ ≥ −0.10 | 3/3 (−0.07, −0.09, −0.05): pass | 3/3 (−0.04, −0.00, −0.03): pass |
| (c) CI vs GS12f > 0 | +0.42 [+0.26, +0.56]: pass | +0.45 [+0.31, +0.60]: pass |
| **D1_k** | **pass** | pass |

- **Recommendation:** X1f-lite = X1f-1, the smallest k that passes.
- On dev, both picks tie X1f: X1f-1 is +0.00 [−0.05, +0.07] and X1f-2 is −0.00 [−0.05, +0.06].
- One single-BEATs subset, B1, has its eval CI just below 0 (−0.09 [−0.16, −0.01]). It still passes (b).

**D2: diminishing returns, under both weightings.** Mean eval Amean at k = 0, 1, 2, 3:

| weighting | k = 0 | k = 1 | k = 2 | k = 3 | step 1→2 | step 2→3 |
|---|---|---|---|---|---|---|
| block | 95.03 | 95.44 | 95.49 | 95.51 | +0.05 | +0.02 |
| member | 95.03 | 95.30 | 95.43 | 95.51 | +0.14 | +0.08 |

- Under block weighting, the first BEATs network brings 85 % of the full gain of +0.49.
- Every later step is under the noise floor.

### What makes one network enough: the block weight

- **Keeping BEATs' total share is what makes one network enough.** At 1.5 per net, one network loses
  about 0.14 more on eval than at 4.5: B0 is −0.22 below X1f instead of −0.07. The ratio picked on
  dev in `grand-ensemble` (r = 0.375 of the from-scratch weight) works for the block, whatever its size.
- This also explains why `ablation/beats`' one-seed additions looked weaker. They added one network
  at the per-net weight (+0.10 to +0.24 vs +0.37 for three), which corresponds to the secondary
  weighting here.

### Per machine, eval (fig2)

- **X1f-1 gives up ToyCar.** It is −0.39 on ToyCar, BEATs' strongest machine, and +0.24 on ToyConveyor.
  It is within ±0.13 elsewhere.
- **X1f-2 is the same, smaller:** ToyCar −0.25, pump +0.15.
- More BEATs networks average out the noise in ToyCar's BEATs scores. The overall loss is small
  because the other machines barely move.

### Predictions

| ID | predicted | got | |
|---|---|---|---|
| Q1 | k=1 mean eval 95.30 [95.15, 95.45] | 95.44 | hit, at the top edge |
| Q2 | k=2 mean eval 95.44 [95.33, 95.53] | 95.49 | hit |
| Q3 | X1f-1 CI lower bound −0.32 [−0.45, −0.18]; D1_1 fails | −0.155; passes | **miss**: I anchored on `ablation/beats`' one-seed additions, made at the per-net weight |
| Q4 | X1f-2 CI lower bound −0.17 [−0.25, −0.08]; D1_2 passes | −0.095; passes | hit |
| Q5 | diminishing returns | yes, both weightings | hit |
| Q6 | member − block, k=1 mean −0.05 [−0.20, +0.10] | −0.14 | hit |
| Q7 | dev pick for k=1 also the best on eval (40 %) | no (B0 picked; B2 best on eval) | miss, as 60 % expected |

## What this means

- **To make X1f cheaper, drop BEATs networks, not machine types.**
  - X1f-lite (one BEATs network, block weight kept) costs 0.45 of X1f's FLOPs and loses 0.07 on eval.
  - `../beats-router`'s R\* costs 0.58 and loses 0.11, with a CI entirely below 0.
  - Routing by machine type fails because dev and eval disagree on where BEATs helps. Dropping
    networks changes nothing per machine, so it doesn't have that problem.
- **Training cost falls too.** A new build needs one BEATs run (~19 h) instead of three.
- **Caveats.**
  - The study is post hoc: all members were seen.
  - There are only 3 BEATs seeds, so "any single seed works" rests on 3 draws.
  - The k = 1 networks are the phase-1 BEATs runs, not fresh ones.

## Figures

| file | shows |
|---|---|
| `plots/fig1_count.png` | dev and eval Amean against the number of BEATs networks (and FLOPs), subsets as dots, means as lines, both weightings. Under block weight, one network gets most of the way to X1f |
| `plots/fig2_machine_delta.png` | per-machine Δ vs X1f for X1f-1 and X1f-2, eval beside dev. The loss is concentrated on ToyCar |

## Reproduce

```bash
bash phase3/beats-count/run.sh links
bash phase3/beats-count/run.sh dev        # commit scores/selection.json BEFORE the next line
bash phase3/beats-count/run.sh eval       # refuses to run unless selection.json is committed and unmodified
/home/f74134118/anaconda3/bin/python phase3/beats-count/make_plots.py
```
