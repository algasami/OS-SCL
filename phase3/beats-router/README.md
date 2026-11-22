# beats-router — run X1f's BEATs block only on some machine types

> **Post hoc by construction (phase-3 rule 2).** No network is trained. The study re-combines the
> 15 networks of X1f, and their eval scores have all been seen. The point estimates are also
> **arithmetic** (rule 3; *Why the point estimates are already known*). The study adds three
> things: bootstrap CIs, how stable a dev-chosen routing is, and how much inference compute it saves.

## Question

X1f (95.19 dev / 95.51 eval, with external pretraining) adds 3 BEATs networks to the 12 from-scratch
networks of GS12f. One BEATs network has 90.95 M parameters; one from-scratch member has 1.38 M. So
the BEATs block is most of X1f's inference cost. The machine type of every clip is known at test
time (DCASE 2020 task 2), so a **hard router on machine type** is free: run BEATs only for the types
in a set M, and score the other types with GS12f alone. This is the one place in X1f where a
mixture-of-experts gate saves compute (discussion with the user, 2026-11-22).

- Does a router picked on dev keep X1f's eval accuracy?
- How much compute does it save?

## Why the point estimates are already known

Every metric is computed within a machine type and then averaged over the 6 types. Turning BEATs
off for type m changes only type m's scores. Routed system R_M therefore equals X1f on the types in
M and GS12f on the rest, machine by machine. Both are in `synthesis/ensemble-g8/scores/ensembles.csv`.
Per-machine BEATs effect, Amean(X1f) − Amean(GS12f):

| | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| dev | −0.03 | **+0.77** | −0.12 | **+0.60** | −0.28 | **+0.17** |
| eval | −0.10 | −0.10 | +0.51 | +2.27 | +0.27 | +0.06 |

The primary router (below) is therefore **R\* = {pump, ToyCar, valve}**: dev 95.26, eval **95.40**
(X1f 95.51, GS12f 95.03). That is −0.11 against X1f on eval. On dev it is +0.07, which is pure
selection optimism, since R\* is the dev optimum over all 64 sets. These numbers are stated now,
not predicted. `score.py` recomputes them from the dumps and asserts they match.

## Setup

- **Members**: exactly X1f's. 8 `fc_nogain` (2060–67) and 4 `fc_spec` (2060–63), S1′ each, weight 1;
  3 BEATs (`bt_arc_s2040–42`), zK′ each, weight 1.5. Machinery: `synthesis/ensemble-g8/score.py`
  (imported; no new scoring rule). Dumps are read from `../OS-SCL-syn-fresh` (scratch).
- **Router R_M**: BEATs weight 1.5 on the types in M, 0 on the rest. There are 64 sets, from
  M = ∅ (GS12f) to M = all six (X1f).
- **Cost** of a system = sum over its members of per-clip inference cost, network forward only. The
  Log-Mel front end, the fbank and the kNN lookup are excluded and listed separately if they are
  measured. A router's cost is **type-weighted**: each machine type counts 1/6, independent of test-set
  clip counts. Clip-weighted cost is reported beside it.
  - primary unit: forward FLOPs per 10-s clip (`torch.utils.flop_counter`, deterministic);
  - secondary: GPU latency on GPU 0, which is idle, fp32, batch 1 and 32, median of 50 runs after warm-up.
    It is never compared across GPUs (CLAUDE.md, GPU 1).
- **CIs**: `grand-ensemble`'s paired clip bootstrap, stratified per machine × ID × label, with the
  same draws and seed.

### Routers (fixed now, dev only)

| name | rule | status |
|---|---|---|
| **R\*** | M = {m : dev Δ_m > 0} | primary; = {pump, ToyCar, valve} |
| R_ci | M = {m : dev paired-bootstrap 95 % CI of Δ_m excludes 0 from above} | secondary; set unknown until computed |
| frontier | all 64 sets, cost vs Amean, dev and eval | descriptive only; picking from it on eval would be test-set selection |

### Decision rules

- **D1 — low-cost variant.** R\* is reported beside X1f as its low-cost variant ("X1f-R") if all three hold:
  - (a) its type-weighted FLOPs are ≤ 70 % of X1f's;
  - (b) the eval paired CI of R\* − X1f has its lower bound ≥ −0.20 Amean. The margin of 0.20 is one
    step above the ~0.15 ensemble noise floor;
  - (c) the eval paired CI of R\* − GS12f excludes 0 from above, so the routed BEATs still add something.

  X1f stays the reported accuracy number whatever D1 says.
- **D2 — accuracy gain.** The router improves accuracy only if the eval CI of R\* − X1f excludes 0
  from above. The point estimate is −0.11, so D2 cannot pass. It is stated so that nobody reads one.
- **D3 — routing is stable on dev.** For each type, resample dev clips (2000 draws, the same resampler)
  and record how often Δ_m > 0. The routing is "stable" if every type is decided in ≥ 90 % of draws, on or off.
  If D3 fails, R\* is described as one draw of an unstable choice, whatever D1 says.

### Descriptive, no decision

- **Cross-base.** Apply R\* to the other two BEATs fusions with the same BEATs networks: XG8 vs G8f, and
  X1 vs GS8 (fresh-confirm). Report sign agreement of Δ_m between dev and eval per machine.
- **Per-ID Amean** for every system beside the pooled numbers.

## Predictions (written before anything is computed; 80 % intervals)

| ID | quantity | prediction |
|---|---|---|
| P1 | BEATs block's share of X1f's forward FLOPs | 95 % [88, 99] |
| P2 | R\* type-weighted FLOPs as a fraction of X1f's | 0.53 [0.50, 0.62] |
| P3 | R\* type-weighted GPU latency (batch 32) as a fraction of X1f's | 0.58 [0.50, 0.75] |
| P4 | eval CI of R\* − X1f, lower bound | −0.26 [−0.36, −0.17] → D1(b) **fails** more likely than not (60 %) |
| P5 | eval CI of R\* − GS12f excludes 0 from above | yes (85 %) |
| P6 | D3: pump and ToyCar are "on" in ≥ 95 % of dev draws; fan in 40–75 %; D3 overall | D3 **fails** (75 %) |
| P7 | R_ci | {pump, ToyCar} (55 %); {pump} (25 %) |
| P8 | cross-base: per-machine sign of Δ_m agrees between dev and eval | 3 of 6 machines per base [2, 4] |

## Results

*(not yet computed)*
