# Pooled kNN — does kNN on the pooled pre-`linear7` feature map add to S1′?

> **Status: DONE 2026-11-05. D1 passes: C1 = S1′ + zGmean replaces S1′ as the
> single-model rule.**
> - **C1** adds the z-scored 1-NN cosine distance of the *global time-mean* of the
>   pre-`linear7` map (512-d) to S1′. It was chosen on seeds 2040–2043 dev (+0.13 Amean).
> - **Confirm dev (seeds 2044–2047):** +0.15 AUC, +0.23 pAUC, **+0.19 Amean**
>   [+0.13, +0.25], 4/4 seeds positive.
> - **Eval (8 seeds):** +0.24 AUC, +0.18 pAUC, **+0.21 Amean** [+0.14, +0.28], 8/8 seeds
>   positive.
> - **Where the gain comes from:** ToyCar (+0.42 dev, +0.77 eval; eval id_05 +2.49) and
>   ToyConveyor (+0.31, +0.98). Eval valve loses 0.35 (id_05 −0.51, 0/8 seeds).
> - **One `ao_nogain` net under C1** scores dev 96.55 / 91.78 / 93.22 (Amean 94.16) and
>   eval Amean 94.09 (8-seed means). S1′ gives 94.00 / 93.88 and S1′ᵀ 94.12 / 93.93.
> - **Every max-pooled term hurts** (C2 −0.33, C5 −0.28 on selection dev). BEAM's
>   mean + max fusion (C3) and its full back-end (C6) land near zero.
> - **Ensemble (secondary):** Σ C1 over G8 against E2′[G8] is +0.01 dev and +0.08 eval
>   Amean, and both CIs include 0.
> - **Diagnostic:** time-mean pooling makes ToyCar scores roll-invariant (0.13 / 0.20 sd
>   at the farthest rolls, against S1′'s 27 / 44 sd). Time-max only *reduces* the effect
>   (1.3 / 2.5 sd). So the time mean of the map before `linear7` carries no absolute position of its
>   own: it enters at `linear7`.
> - Pre-registered at 77f2598; selection committed at ed089e8 before any confirm-seed or
>   eval score under a new rule.

## Why this exists

The adopted single-model rule is S1′ = zR0′ + zK′ (`../ensemble-backend`,
`../confirm-seeds`): the ArcFace CE of the clip's own ID, plus the 1-NN cosine
distance K of the 128-d pre-head embedding to that ID's training clips, each
z-scored per ID with training-part statistics. Both terms see only one vector,
`linear1(linear7(map))`.

`linear7` (`model/net.py`) is a depthwise convolution with an (8, 20) kernel over
MobileFaceNet's final 512 × 8 × 20 map (frequency × time): a **learned weighting
of every time position**, per channel. `../tta`'s pilot showed the ToyCar models
key on *absolute* time position: a circular roll of half a clip raises normal
ToyCar R0 by 56–69 sd while fan, pump and slider do not move. `linear7` is a
plausible place for that dependence to live.

**Literature.** BEAM / AdaBEAM (arXiv 2603.13749, TASLP 2026) kept a fine-tuned
BEATs fixed and changed only the kNN back-end: fusing a temporal-**mean**-pooled
and a temporal-**max**-pooled kNN score, and adding **sub-band** kNN (each
frequency row pooled over time, one kNN per band, combined). On DCASE 2020 that
raised dev Amean 91.6 → 95.6 and eval 93.0 → 96.2, most of it from the back-end.

**Hypothesis.** kNN on the map *before* `linear7`, pooled over time (mean and
max) globally and per frequency band, carries information that the ArcFace
logit and the 128-d embedding miss. Pooling over time also discards absolute
position, which may matter for ToyCar.

**How this differs from `../memory-bank`.** That ablation kept every one of the
8 × 20 positions as its own memory-bank entry and scored a clip by its worst
patch (PatchCore). Every patch arm was 13–40 dev AUC below clip scoring on its
own and below S1′ when fused (−0.72 dev, −0.22 eval). Here nothing is scored
per position: every vector is *pooled over all 20 time steps*, so each clip
contributes one vector (global) or one vector per band (8 bands), as K does.

## What has already been seen

- **The 8 `ao_nogain` checkpoints** (seeds 2040–2047, `../aug-ops`) under R0 and
  S1′ on dev and eval, per seed (`../aug-ops/scores/s1p.csv`). 8-seed S1′: dev
  96.43 / 91.58 / 92.82 (Amean 94.00), eval Amean 93.88.
- Their pre-head embeddings and CE (`../ensemble-backend/extract.py` dumps).
- `../tta`'s pilot (roll vs R0 on held-out training clips) and its S1′ᵀ results.
- **Not seen:** any statistic of the pre-`linear7` map, on any clip.

## Design

### Features

Per clip, the output of `mobilefacenet.conv2` (1×1 conv + BN + PReLU, the input
to `linear7`), shape 512 × 8 × 20 (channels × frequency × time; asserted at
extraction). Stored, never the full map:

| name | shape | definition | stored as |
|---|---|---|---|
| `bmean` | 8 × 512 | per frequency band, mean over the 20 time steps | fp16 |
| `bmax` | 8 × 512 | per frequency band, max over the 20 time steps | fp16 |
| `gmean` | 512 | mean over all 8 × 20 positions (= mean of `bmean` over bands) | fp32 |
| `gmax` | 512 | max over all 8 × 20 positions (= max of `bmax` over bands) | fp32 |

Plus R0 and the 128-d pre-head embedding from the same forward pass, so S1′ is
recomputed from the new dump and checked against the old one.

**"Global" pools over time *and* frequency**, BEAM's global pooled vector;
"band" keeps the 8 frequency rows apart and pools over time only. fp16 keeps a
model's dump at ~1 GB instead of ~16 GB for the full map. The global vectors are
kept in fp32 so the fp16 rounding can be checked: `gmean` recomputed from fp16
`bmean` must give the same per-ID AUC (P0).

### Scores

All distances are **1-NN cosine distances** to the ID's training-part clips (the
95% the checkpoint trained on, replayed from its seed exactly as `../ensemble-backend`
does), on L2-normalised, **uncentred** vectors — the same metric as K.

- `Gmean`, `Gmax`: 1-NN distance of `gmean` / `gmax`.
- `Bmean`, `Bmax`: per band b, the 1-NN distance d_b of that band's vector to the
  same band of the bank; each d_b is z-scored per ID, and the band score is the
  **mean over the 8 bands** of those z-scores.
- **Every score is z-scored per ID with training-part statistics, exactly as
  zK′:** mean and sd over the ID's training-part clips, each scored
  leave-one-out against the rest of the bank. A composite (`Bmean`, `Bmax`,
  `Gmm`, `All`) is built from z-scores and **re-z-scored** the same way, so every
  candidate adds one unit-variance term.
  - `Gmm` = zGmean + zGmax (BEAM's mean + max fusion).
  - `All` = zGmean + zGmax + zBmean + zBmax (BEAM's full back-end analogue).

### Candidate rules (fixed; exactly six)

| rule | definition | what it tests |
|---|---|---|
| C1 | S1′ + zGmean | uniform instead of learned time weighting, 512-d |
| C2 | S1′ + zGmax | max over the map: transients, no absolute position |
| C3 | S1′ + zGmm | BEAM's mean + max fusion |
| C4 | S1′ + zBmean | per-band (sub-band) kNN, time-mean |
| C5 | S1′ + zBmax | per-band kNN, time-max |
| C6 | S1′ + zAll | BEAM-style full back-end, one unit-weight term |

S1′ = zR0′ + zK′ throughout, recomputed with `../ensemble-backend/score.py`'s
`per_run_scores` (imported unchanged). The standalone z-scores (zGmean, zGmax,
zGmm, zBmean, zBmax, zAll, next to zR0′ and zK′) are reported as descriptives
only; none of them is a candidate.

### Selection and confirmation protocol

1. **Stage A — selection (dev, seeds 2040–2043 only).** Extract train bank + dev
   clips for all 8 seeds; **score only seeds 2040–2043**. The chosen rule is the
   candidate with the highest 4-seed mean dev Amean Δ vs S1′
   (Amean = mean(AUC, pAUC), each averaged over the 6 machines). The choice is
   committed before anything else is scored.
2. **Stage B — confirmation.** Then, and only then: dev of seeds 2044–2047 under
   the chosen rule, and eval of all 8 seeds (eval clips are not even extracted
   before the selection commit). The other five candidates are reported on the
   confirm cells too, as descriptives, after the chosen one.
3. **Uncertainty:** a paired clip-bootstrap (2000 stratified resamples, seed 0,
   `../ensemble-backend`'s resampler), of the seed-mean Δ, the same resample for
   both rules.
4. **Secondary — ensemble:** E2′[G8] = Σ over the 8 `ao_nogain` seeds of S1′,
   against Σ of the chosen rule. Reported, not a decision.

### Diagnostic — does time pooling remove ToyCar's position dependence?

ToyCar test clips (dev and eval; all 8 seeds), each circularly rolled by k/8 of
the clip, k = 0…7 (the wave is rolled on the GPU and the mel recomputed from it,
as in `../tta`; k = 0 uses the CPU mel). The bank and every z-statistic stay
**unrolled**. Scores: S1′, zGmean, zGmax and the chosen rule. Run in stage B
(it scores eval ToyCar under the new features).

- **Penalty(k)** per ID: mean over the ID's *normal* test clips of
  s(roll k) − s(roll 0), in units of the sd of s(roll 0) over those clips; then
  averaged over IDs and seeds. The headline is the mean over the three farthest
  rolls (k = 3, 4, 5, i.e. 37.5–62.5%).
- **AUC(k):** ToyCar AUC with every test clip rolled by k (bank unrolled).
- **Verdict, fixed now** (per set, on the headline penalty of zGmax):
  - *removes* the position dependence: ≤ 1 sd;
  - *reduces* it: ≤ 25% of S1′'s headline penalty;
  - *does not*: otherwise.

## Predictions (written 2026-11-05, before any pre-`linear7` statistic exists)

Δ = rule − S1′, seed-mean, in points. Intervals are 80%.

- **P0, reproduction.** S0 recomputed from the new dump matches eval.py's per-ID
  AUC to < 5e-4 for every seed and set; the new R0 and pre-head embedding match
  the `../aug-ops` dumps (max |ΔR0| ≤ 1e-3; S1′ overall within 0.01 of
  `s1p.csv`); `gmean` from fp16 `bmean` gives per-ID AUC within 1e-3 of fp32
  `gmean`. If any fails, stop and fix before selecting.
- **P1, standalone.** zGmean dev AUC (4 selection seeds) within ±1.5 of zK′
  (central −0.5). zGmax 1–4 points below zGmean (central −2): max pooling over a
  post-PReLU map is noisy. Per-band scores within ±1 of the matching global.
- **P2, selection.** The best candidate's selection-seed dev Amean Δ is
  +0.10, interval [−0.05, +0.35]. At least one candidate is > 0 (75%).
  Which one wins: C3 or C6 (50%), C1 (25%), any other (25%).
- **P3, confirm dev (seeds 2044–2047), chosen rule.** Δ AUC +0.05
  [−0.15, +0.25]; Δ pAUC +0.10 [−0.20, +0.40]; Δ Amean +0.07 [−0.15, +0.30].
  Winner's curse: the confirm Δ is smaller than the selection Δ (70%).
- **P4, eval (8 seeds), chosen rule.** Δ Amean +0.05 [−0.25, +0.30]. Δ ≥ −0.05
  (60%).
- **P5, where it moves (dev, all 8 seeds, chosen rule).** ToyCar has the largest
  positive Δ AUC of the six machines (40%). ToyConveyor |Δ| ≤ 0.5 (60%). Fan,
  slider and valve each |Δ| ≤ 0.15. Eval ToyCar id_05 (~82 AUC): Δ > 0 (55%).
- **P6, ensemble.** Σ chosen − E2′[G8] on eval Amean: 0.00 [−0.20, +0.15].
  Ensembling already averages away most of what a second kNN view adds.
- **P7, diagnostic.** S1′'s headline ToyCar penalty is ≥ 10 sd on dev and on eval
  (central 30 sd). zGmax's is ≤ 25% of that (central 2 sd, interval [0.3, 10]).
  Verdict: *reduces* (55%), *removes* (30%), *does not* (15%). zGmean behaves
  like zGmax (penalty within a factor of 2). AUC(k): S1′'s ToyCar AUC falls by
  ≥ 5 points at the farthest rolls on dev; zGmax's moves by ≤ 2.

  Caveat written now: neither pooled vector is exactly roll-invariant. Every
  grid cell has a receptive field of a large fraction of the clip, the convs
  zero-pad at the clip edges, the roll moves the splice into the clip, and
  Tgram's `LayerNorm(313)` has a per-frame affine. Mean pooling removes
  absolute position as much as max pooling does; the diagnostic tests both.

## Decision rules

- **D0.** If no candidate has a selection-seed dev Amean Δ > 0, nothing is
  adopted. The argmax is still carried to stage B and reported, but D1 cannot
  pass.
- **D1, adopt the chosen rule as the single-model rule in place of S1′.** All of:
  - selection-seed dev Amean Δ > 0 (it was chosen for this);
  - confirm seeds (2044–2047), dev: the paired-bootstrap 95% lower bounds of
    Δ AUC **and** Δ Amean are both > 0;
  - eval, 8 seeds: Δ Amean ≥ −0.05 (point estimate; the tolerance).
- **D2, ToyCar.** The diagnostic verdict is reported whatever D1 says. It is
  evidence about *where* the position dependence lives, not a rule change.
- The ensemble result (P6) never changes D1. mAUC is reported but single-model
  mAUC deltas under ~2 points are not read (`CLAUDE.md`).
- **Not tested here: composition with TTA.** `../tta` adopted S1′ᵀ (±1% TTA,
  +0.05 eval). This ablation compares against plain S1′, the rule its candidates
  extend. If D1 passes, S1′ᵀ + the new term is the next thing to confirm.

## Stage A — selection (2026-11-05, before any confirm-seed or eval score under a new rule)

Scored by `score.py select` (`scores/scorecard_select.txt`, `select.json`,
`select_metrics.csv`, `p0_select.json`). Eval clips were not extracted yet.

- **P0 passes.** S0 per-ID dev AUC matches eval.py to 1.4e-4 (all 8 seeds). The
  new R0 is bit-identical to the `../aug-ops` dumps on training clips, ≤ 2.4e-4 on
  test clips (the last, differently sized batch, as in `../tta`). S1′ recomputed
  matches `s1p.csv` to < 1e-4. `gmean` / `gmax` rebuilt from the fp16 band arrays
  give per-ID AUC within 2.0e-4 of the fp32 vectors.
- **Chosen: C1 = S1′ + zGmean** (global time-mean pooled kNN), selection-seed dev
  Amean Δ **+0.130** (per seed +0.21, +0.10, +0.08, +0.14). C4 (per-band
  time-mean) is a close second at +0.123. Every max-pooled candidate is negative
  (C2 −0.33, C5 −0.28) and drags the mixtures (C3, C6) below zero.
- Standalone (4 seeds, dev AUC): zGmean 93.40, zBmean 94.43, zGmax 89.33,
  zBmax 90.18, against zK′ 95.54 and R0 96.17.

Stage B (eval extraction, confirm seeds, ToyCar rolls) runs after this commit.

## Results (2026-11-05)

Full output: `scores/scorecard_select.txt`, `scorecard_confirm.txt`,
`scorecard_diag.txt`. Machine-readable files: `confirm.json` (Δs, CIs, D1), `diag.json`,
`*_metrics.csv` (every run × rule × machine) and `confirm_perid.csv`.

### Prediction check

| prediction | outcome | held? |
|---|---|---|
| P0 reproduction | S0 vs eval.py per-ID ≤ 1.9e-4 (dev and eval); R0 bit-identical on train, ≤ 2.4e-4 on test; S1′ vs `s1p.csv` < 1e-4; fp16 bands → global ≤ 2.0e-4 per-ID AUC | **yes** |
| P1 zGmean within ±1.5 of zK′ (dev AUC) | 93.40 vs 95.54 (−2.14) | **no** |
| P1 zGmax 1–4 below zGmean | 89.33, i.e. 4.07 below | **no** (just) |
| P1 per-band within ±1 of global | mean +1.03, max +0.85 | **no** (mean, just) |
| P2 best selection Δ +0.10 [−0.05, +0.35] | +0.130 (C1) | **yes** |
| P2 some candidate > 0 (75%) | C1 and C4 | **yes** |
| P2 winner C3/C6 (50%), C1 (25%) | C1. C3 −0.06, C6 −0.05 | C1 branch |
| P3 confirm dev Δ AUC +0.05 [−0.15, +0.25] | +0.145 | **yes** |
| P3 Δ pAUC +0.10 [−0.20, +0.40] | +0.229 | **yes** |
| P3 Δ Amean +0.07 [−0.15, +0.30] | +0.187 | **yes** (above central) |
| P3 winner's curse: confirm < selection (70%) | +0.187 > +0.130 | **no** |
| P4 eval Δ Amean +0.05 [−0.25, +0.30]; ≥ −0.05 (60%) | +0.207 | **yes** |
| P5 ToyCar largest dev Δ AUC (40%) | ToyCar +0.42 is the largest | **yes** |
| P5 ToyConveyor \|Δ\| ≤ 0.5 dev (60%) | +0.31 | **yes** |
| P5 fan, slider, valve \|Δ\| ≤ 0.15 dev | −0.04, +0.01, +0.06 | **yes** |
| P5 eval ToyCar id_05 Δ > 0 (55%) | +2.49, 8/8 seeds | **yes** |
| P6 ensemble eval Δ 0.00 [−0.20, +0.15] | +0.077 [−0.010, +0.169] | **yes** |
| P7 S1′ headline ≥ 10 sd | 27.1 dev, 44.1 eval | **yes** |
| P7 zGmax ≤ 25% of S1′; central 2 [0.3, 10] | 1.34 (5%) dev, 2.51 (6%) eval | **yes** |
| P7 verdict *reduces* (55%) | *reduces* on both sets | **yes** |
| P7 zGmean within ×2 of zGmax | 0.13 / 0.20 sd: 10× below zGmax; *removes* | **no** |
| P7 S1′ ToyCar AUC −≥5 at far rolls; zGmax moves ≤ 2 | S1′ 98.0 → 54.1 dev; zGmax 85.8 → 79.6 (−6.2) | half: S1′ yes, zGmax **no** |

### D1: adopt

| condition | value | pass |
|---|---|---|
| selection dev Δ Amean > 0 | +0.130 | yes |
| confirm dev Δ AUC, 95% lower bound > 0 | +0.145 [+0.080, +0.216] | yes |
| confirm dev Δ Amean, 95% lower bound > 0 | +0.187 [+0.126, +0.252] | yes |
| eval 8-seed Δ Amean ≥ −0.05 | +0.207 [+0.138, +0.277] | yes |

**C1 replaces S1′ as the single-model rule.** Every confirm-dev seed is positive
(+0.25, +0.14, +0.21, +0.15 Amean), and so is every eval seed (+0.09 to +0.30). mAUC
moves +0.4 to +0.6 but is not read (a single-model mAUC delta under 2 points).

8-seed means, AUC / pAUC / mAUC / Amean (paper: dev 95.71 / 90.23 / 91.23, eval Amean 93.22):

```
            dev                           eval Amean
R0     96.10  91.08  92.12  93.59           93.13
S1′    96.43  91.58  92.82  94.00           93.88
C1     96.55  91.78  93.22  94.16           94.09
```

`../tta`'s S1′ᵀ (±1% TTA on S1′) gave dev 96.52 / 91.72 / 93.01 and eval Amean 93.93.
C1 is above it on every cell, without TTA. **C1 + TTA was not tested.** TTA averages
the pre-head embedding and R0; how it should treat the pooled map is an open design
choice. That is the next confirmation, not something to assume.

### What else the data shows

- **The time-mean pooled view adds what neither existing term has.** On its own it
  is 2.1 AUC below K′ and 2.8 below R0. Added as a third z-term it still gains on 12
  of 12 confirm cells (4 dev seeds, 8 eval seeds). Per-band time-mean (C4) behaves
  almost the same: +0.12 selection, +0.15 confirm dev, +0.14 eval. Keeping the 8
  frequency rows apart adds nothing over the global mean.
- **Max pooling is the wrong half of BEAM here.** Time-max kNN alone is about 4 AUC
  below time-mean (global and per-band), and every rule containing it is below the mean-only rule. On BEATs, BEAM
  found the two complementary. On this post-PReLU map, the max over 160 positions
  looks dominated by edge and splice cells (see the diagnostic).
- **Per machine (8 seeds, Δ AUC, dev / eval):** ToyCar +0.42 / +0.77,
  ToyConveyor +0.31 / +0.98, valve +0.06 / −0.35, and fan, pump and slider all within
  ±0.05.
  - The ToyCar gain is concentrated in the weakest IDs: dev id_01 +1.64 (8/8 seeds)
    and eval id_05 +2.49 (85.1 → 87.6, 8/8).
  - On ToyConveyor, eval id_05 gains +1.23 and id_06 +1.43, both 8/8.
  - The eval valve loss is id_05 (−0.51, 0/8 seeds). It is the one consistent
    per-ID loss.
- **Ensemble.** Σ C1 over the 8 `ao_nogain` seeds against E2′[G8]: dev 94.97 → 94.98,
  eval 94.87 → 94.95 Amean. Both CIs include 0. As with TTA, most of what a second
  view adds to one network, the ensemble already has. No ensemble rule changes here.

### Diagnostic: ToyCar under circular rolls

Every ToyCar test clip is rolled by k/8 of the clip. The bank and all z-statistics stay
unrolled. Penalty = rise of normal clips' score, in sd of the unrolled score; 8-seed
means; headline = mean over k = 3, 4, 5.

| score | dev headline | eval headline | dev AUC k=0 → k=4 | eval AUC k=0 → k=4 |
|---|---|---|---|---|
| S1′ | 27.1 sd | 44.1 sd | 98.0 → 54.1 | 95.2 → 70.4 |
| zR0′ | 26.2 | 49.7 | 97.5 → 55.6 | 94.1 → 69.8 |
| zK′ (128-d embedding) | 23.8 | 25.1 | 98.0 → 54.0 | 95.2 → 71.6 |
| **zGmean (time-mean)** | **0.13** | **0.20** | 95.0 → 94.5 | 95.3 → 94.8 |
| zGmax (time-max) | 1.34 | 2.51 | 85.8 → 79.9 | 90.6 → 78.1 |
| C1 | 24.4 | 35.6 | 98.4 → 56.1 | 96.0 → 71.6 |

- **The pre-`linear7` map is roll-equivariant.** Its time mean barely moves when the
  clip is rolled half way: 0.13 sd, against 27 sd for S1′. Everything after
  `mobilefacenet.conv2` is `linear7`, a 1×1 conv and the head. Of those, only
  `linear7`, the learned (8, 20) per-position weighting, depends on position. So
  **ToyCar's absolute-time dependence enters at `linear7`**, and both R0 and K′
  inherit it. It does not come from Tgram's per-frame `LayerNorm` affine, the
  zero-padding or the splice; those would show up in the pooled map.
- **Max pooling only reduces it.** The verdict is *reduces* on both sets (1.3 and
  2.5 sd), and zGmax's ToyCar AUC drops 6–13 points under a roll. A max over 160
  cells is sensitive to the few cells that a roll actually changes: the splice and
  the cells that were at the clip edges. A mean over 20 time steps dilutes them.
  So the brief's premise is backwards here: max pooling discards absolute position
  less completely than mean pooling does.
- **C1 is not position-invariant.** Its penalty is 24 / 36 sd, because S1′'s two
  terms still carry the position weighting. C1 gains on ToyCar by adding a
  position-free view next to them, not by removing the dependence.
- **Position helps some ToyCar IDs and hurts others** (`scores/toycar_perid.csv`,
  8-seed means, unrolled).
  - At k = 0 the position-free zGmean scores 95.0 on dev ToyCar against S1′'s 98.0,
    and 95.3 against 95.2 on eval.
  - Per ID it splits. On dev id_03, position is what works: zGmean 88.6 against
    S1′ 99.7.
  - On the IDs where S1′ is weakest, the position-free score is the better one:
    eval id_05 87.3 vs 85.1, and dev id_01 93.1 vs 92.5.
  - C1 keeps both (id_03 99.6, id_05 87.6, id_01 94.1). This is consistent with
    the recording position mattering only for some IDs, but that is not tested here.

### Incidents (none changes a score)

- **GPU sharing.** Between stage A and stage B, the `permix` runner started a
  300-epoch training on GPU 1 and took its lock. Stage B (≈ 15 min, ≈ 1.4 GB) ran
  next to it rather than waiting hours: `PK_NOLOCK=1` in `run.sh`. Extraction is
  deterministic in eval mode, so sharing changes only wall-clock.
- **Corrupt dumps on disk (faulty host RAM).** Three ToyCar roll dumps failed their
  zip CRC on read (`np.load` checks it), although each had passed the in-RAM range
  check before `np.savez`. They stayed bad after page-cache eviction, so the bytes on
  disk were corrupt. A re-extraction wrote one bad file again.
  - `extract.py` now saves through `save_verified`: write, fsync, evict from the page
    cache, CRC-test from disk, and rewrite on failure.
  - Every dump in `results/emb` passed `testzip` before the diag stage ran.
  - The extractor also aborted (core dump) twice on startup, and the retry loop
    reran it. Both are the hardware failure modes `CLAUDE.md` describes.

## Figures

`/home/f74134118/anaconda3/bin/python ablation/pooled-knn/make_plots.py` reads `scores/` only.

| figure | shows |
|---|---|
| `plots/fig1_selection.png` | selection-seed dev Δ vs S1′ per candidate (dots = seeds), plus standalone dev AUC. C1 and C4 are positive; every max-pooled term is negative |
| `plots/fig2_confirm.png` | C1 − S1′ on confirm dev and eval with paired 95% bootstrap and per-seed dots (AUC, pAUC, Amean), plus the G8 ensemble Δ, whose CIs include 0 |
| `plots/fig3_per_machine.png` | C1 − S1′ AUC per machine, dev and eval: ToyCar and ToyConveyor gain, eval valve −0.35 |
| `plots/fig4_toycar_roll.png` | ToyCar normal-clip penalty and AUC vs circular roll for S1′, zGmean, zGmax and C1. Time-mean is flat; S1′ rises 27–44 sd |

## What this cannot answer

- **One recipe.** Every number is `ao_nogain`. C1 is adopted as the rule to
  pre-register for the next new checkpoints, not shown to transfer to other recipes.
- **Not composed with TTA** (see D1 above), nor tried inside the A8 ensemble.
- **Eval valve id_05** loses consistently (−0.51, 0/8). It is not understood.
- **The `linear7` attribution rests on the roll test alone.** No intervention was run,
  such as replacing `linear7` with a uniform mean and retraining.

## Files

| file | what |
|---|---|
| `extract.py` | GPU: per-clip pooled pre-`linear7` features + R0 + pre-head embedding; `--split train/dev/eval`, `--rolls` for the diagnostic; CRC-verified writes |
| `score.py` | CPU: P0 checks, candidates, selection / confirm / diagnostic stages, bootstrap, scorecard |
| `run.sh` | stages: `extractA`, `select`, `extractB` (refuses until `select.json` is committed), `confirm`, `diag`, `toycar`, `plots`; resumable; `PK_NOLOCK=1` skips the GPU lock |
| `make_plots.py` | figures from `scores/` only |
| `scores/` | committed record: `select*`, `confirm*`, `diag*`, `toycar_perid.csv`, `p0_*.json`, `scorecard_*.txt` |
| `results/` | dumps (~9 GB) and logs: gitignored scratch |
