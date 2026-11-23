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

> **Verdict: D1 fails, by 0.02 on (b). D2 and D3 fail. There is no "X1f-R".** On dev, the router
> picked R\* = {pump, ToyCar, valve}. It saves 42 % of X1f's forward FLOPs. On eval it is worse
> than X1f, −0.11 [−0.22, −0.03]. The clip-bootstrap CI is entirely below zero, so for these 15
> networks it is a loss, not a tie. Its size is under the ~0.15 seed-noise floor (phase-3 rule 4),
> so it says nothing about how large the loss would be for a fresh build. It still beats GS12f,
> +0.37 [+0.28, +0.47]. **X1f stays the system with external pretraining.**

`score.py` reproduced every system against `ensemble-g8` to ≤ 4.4e-5 Amean. Separability holds
exactly: per-machine metrics of all 64 routers equal their base system's, with a difference of 0.
R\* built directly equals the mixed version. Scoring was replicated byte for byte (tries 1 and 2).

### Cost (`scores/cost.json`; RTX A4000, GPU 0, torch 2.6, network forward only)

| member | FLOPs / clip | params | latency, batch 32, per clip |
|---|---|---|---|
| `fc_nogain` | 4.50 G | 1.38 M | 1.46–1.49 ms |
| `fc_spec` | 4.68 G | 2.96 M | 1.52–1.55 ms |
| BEATs | 89.93 G | 90.91 M | 15.2–16.9 ms |

- Latency ranges span three runs: `scores/cost.json` is run 3, and runs 1–2 are in `scores/cost_runs/`.
  FLOPs and parameter counts are identical in all three runs.
- BEATs measures 90.91 M parameters; the Question's 90.95 M was an estimate.
- The BEATs block is **83.1 % of X1f's 324.5 GFLOPs** and 72–74 % of its batch-32 latency.
- Front ends are excluded. On CPU, the Log-Mel costs 0.8–1.1 ms per clip and the BEATs fbank 1.1–1.4 ms
  per clip across the three runs.

R\* cost as a fraction of X1f:

| measure | fraction |
|---|---|
| FLOPs, type-weighted (D1(a)) | **0.584** |
| FLOPs, clip-weighted | 0.489 dev / 0.550 eval |
| latency, batch 32 | 0.632 |
| latency, batch 1 | 0.72 (0.67–0.73 over three runs; batch-1 timing of the from-scratch nets is unstable) |

### Systems and decisions (Amean, paired 95 % CI)

| system | dev | eval | per-ID dev / eval | vs X1f, eval | vs GS12f, eval |
|---|---|---|---|---|---|
| X1f (BEATs everywhere) | 95.19 | 95.51 | 95.35 / 95.35 | — | +0.49 |
| **R\*** = R_ci (BEATs on pump, ToyCar, valve) | 95.26 | 95.40 | 95.38 / 95.20 | **−0.11 [−0.22, −0.03]** | +0.37 [+0.28, +0.47] |
| GS12f (no BEATs) | 95.01 | 95.03 | 95.15 / 94.80 | −0.49 | — |

- Per-ID Amean (the literature's convention) puts R\* − X1f at −0.16 on eval, larger than the pooled −0.11.
- D1(b) misses by 0.021. The Monte Carlo SE of the 2.5th percentile at 2000 draws is ≈ 0.003, so this is a clear miss.

| rule | needs | got | result |
|---|---|---|---|
| D1(a) | FLOPs ≤ 0.70 of X1f | 0.584 | pass |
| D1(b) | eval CI lower bound of R\* − X1f ≥ −0.20 | **−0.221** | **fail** |
| D1(c) | eval CI of R\* − GS12f > 0 | [+0.28, +0.47] | pass |
| D2 | eval CI of R\* − X1f > 0 | [−0.22, −0.03] | fail (known in advance) |
| D3 | every type decided in ≥ 90 % of dev draws | fan 26 %, slider 19 % | **fail** |

- R_ci came out equal to R\*: valve's dev CI lower bound is +0.03, just above 0.
- R\* is the best of the 64 routers on dev (+0.07 over X1f, the selection optimism). It ranks 28th of 64 on eval.

### Why it fails: dev does not predict which machines BEATs helps

Per-machine BEATs effect Δ_m = X1f − GS12f, and the share of 2000 resamples with Δ_m > 0:

| | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| dev Δ [95 % CI] | −0.03 [−0.12, +0.05] | +0.77 [+0.40, +1.19] | −0.12 [−0.40, +0.03] | +0.60 [+0.41, +0.81] | −0.28 [−0.57, −0.03] | +0.17 [+0.03, +0.39] |
| dev draws > 0 | 26 % | 100 % | 19 % | 100 % | 1 % | 100 % |
| eval Δ | −0.10 | −0.10 | +0.51 | +2.27 | +0.27 | +0.06 |
| eval draws > 0 | 2 % | 27 % | 100 % | 100 % | 98 % | 76 % |

- **The dev and eval signs disagree on 3 of 6 types. Only ToyConveyor is confident on both sides.**
  - ToyConveyor's dev CI is entirely below 0, yet eval puts it above 0 in 98 % of draws.
  - Pump is confident only on dev: "on" in 100 % of dev draws, "off" in 73 % of eval draws.
  - Slider is confident only on eval.
- **This is not clip-resampling noise.** The clip bootstrap is confident on both sides for ToyConveyor.
  - Dev and eval test sets hold disjoint machine IDs (fan dev 00/02/04/06, eval 01/03/05).
  - A plausible reason is that BEATs' benefit varies by ID. That was not tested here: no per-ID Δ was computed.
  - The bootstrap is stratified by ID, so it cannot see ID-to-ID variance. D3 therefore understates how unstable the routing is.
- **A resampled dev picks exactly R\* in 60 % of draws.** The eval-optimal set {slider, ToyCar, ToyConveyor, valve} never comes up on dev.
- **The pattern repeats with the two other bases** (descriptive). They share the same 3 BEATs networks and a subset of GS12f's members, so this is not an independent replication. Dev and eval signs agree on 3 of 6 machines for each of the three bases. Routing with R\* on eval:

  | pair | R\* routed, eval | vs BEATs everywhere |
  |---|---|---|
  | X1f vs GS12f | 95.40 | −0.11 |
  | XG8 vs G8f | 95.38 | −0.07 |
  | X1 vs GS8 | 95.29 | −0.13 |

- **ToyCar carries most of the BEATs gain on eval and about half on dev.** On eval it is 0.38 of the +0.49. On dev it is 0.10 of the +0.18, where pump adds 0.13.
  - BEATs on ToyCar alone costs 0.31 of X1f's FLOPs and scores 95.40 on eval, the same as R\*. On dev it is 95.11, below R\*'s 95.26.
  - ToyCar is the one type where dev and eval agree strongly. The rule was fixed on dev, so this is a description, not a pick (*What next*).

### Predictions

| ID | predicted | got | |
|---|---|---|---|
| P1 | BEATs share of FLOPs 95 % [88, 99] | 83.1 % | **miss**: a from-scratch member costs 4.5 GFLOPs, so BEATs is 20× the FLOPs, not 66× like its parameter count |
| P2 | R\* FLOPs 0.53 [0.50, 0.62] | 0.584 | hit |
| P3 | R\* latency, batch 32, 0.58 [0.50, 0.75] | 0.632 | hit |
| P4 | eval CI lower bound −0.26 [−0.36, −0.17]; D1(b) fails | −0.221; fails | hit |
| P5 | R\* − GS12f eval CI > 0 | [+0.28, +0.47] | hit |
| P6 | pump, ToyCar ≥ 95 %; fan 40–75 %; D3 fails | 100 %, 100 %; fan 26 %; fails | fan part **miss**, rest hit |
| P7 | R_ci = {pump, ToyCar} or {pump} | {pump, ToyCar, valve} | **miss** |
| P8 | 3 of 6 sign agreements per base [2, 4] | 3 / 3 / 3 | hit |

## What this means for the MoE suggestion

- **A machine-type router is the one gate in X1f that is free,** since machine type is known at test
  time. It saves real compute (42 % of FLOPs), but a gate fit on dev picks the wrong machines for eval.
  The cost is 0.11 eval Amean, and the clip-bootstrap CI excludes zero.
- **Any gate selected with dev anomaly labels inherits this dev → eval ID shift.** A clip-level gate
  trained the same way would too. Two other options were not tested:
  - a gate driven by per-ID normal-only signals (the eval IDs' normal training clips exist);
  - other compute-saving schemes, such as confidence cascades.
- **To make X1f cheaper, shrink the BEATs block itself, not its routing:** fewer BEATs networks, or
  distil them. That was not tested here.

## What next

1. **ToyCar-only BEATs** (0.31 of X1f's FLOPs) is the obvious follow-up. It would be chosen after
   seeing eval here, so it needs fresh networks: new BEATs seeds (~19 h each) and/or new
   from-scratch seeds, with a pre-registration written first.
2. **One BEATs network instead of three** at the same block weight is a cost cut that does not route
   by machine type. It needs no training: score it on the existing dumps, pre-registered.

## Figures

| file | shows |
|---|---|
| `plots/fig1_machine_delta.png` | per-machine BEATs effect Δ_m: dev with 95 % CI, eval as points; R\* types shaded. The sign flips between sets on pump, slider and ToyConveyor |
| `plots/fig2_frontier.png` | all 64 routers, type-weighted FLOPs vs Amean, dev and eval. R\* is the dev optimum and sits mid-pack on eval |
| `plots/fig3_stability.png` | D3: share of resamples with Δ_m > 0 per machine, dev bars and eval points. Fan and slider are undecided on dev |

## Reproduce

```bash
bash phase3/beats-router/run.sh links     # symlink tree to ../OS-SCL-syn-fresh dumps -> results/emb
bash phase3/beats-router/run.sh score     # replicated until two tries are byte-identical -> scores/
/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python phase3/beats-router/cost.py   # GPU 0
/home/f74134118/anaconda3/bin/python phase3/beats-router/make_plots.py
```
