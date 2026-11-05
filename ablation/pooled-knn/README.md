# Pooled kNN — does kNN on the pooled pre-`linear7` feature map add to S1′?

> **Status: stage A done 2026-11-05 — C1 (S1′ + zGmean) chosen on seeds 2040–2043 dev;
> confirmation pending.** Design, candidates, protocol, predictions and decision rules
> were committed at 77f2598, before any new score existed.

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

## Files

| file | what |
|---|---|
| `extract.py` | GPU: per-clip pooled pre-`linear7` features + R0 + pre-head embedding; `--split train/dev/eval`, `--rolls` for the diagnostic |
| `score.py` | CPU: P0 checks, candidates, selection / confirm / diagnostic stages, bootstrap, scorecard |
| `run.sh` | stages: `extractA`, `select`, `extractB`, `confirm`, `diag`, `plots`; resumable |
| `make_plots.py` | figures from `scores/` only |
| `scores/` | committed record |
| `results/` | dumps (~9 GB) and logs: gitignored scratch |
