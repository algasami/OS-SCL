# Literature arms: three published add-ons, each at n=4 on the fast recipe

> **Status: stage 1 done (2026-10-26); D2 replication of `aug` next.**
> `aug` (waveform augmentation) passes D1: +0.84 dev AUC [+0.45, +1.25],
> 4/4 seeds, but only +0.20 eval [−0.20, +0.59]. `gate` and `eca` fail on dev
> (−0.02, −0.04). Under S1′, `aug` beats all four paper numbers on 4/4
> seeds; that is not a claim until the fresh-seed replication. Pre-registered 2026-10-17. A pre-launch addendum
> (2026-10-23, below Decision rules) fixes the base and records the measured
> noise. The predictions and rules are unchanged.

## Why this exists

A single network still trails the paper on dev: 95.36 dev AUC under S1′
against 95.71, and about 1 point short on mAUC (`../confirm-seeds`). A
literature survey (2026-10-17) found three cheap, well-specified add-ons with
ablation evidence on DCASE 2020 Task 2 dev. None of them has been tested on
top of OS-SCL's mixup + SupCon recipe:

| arm | what | source | reported gain (dev AUC), and over what |
|---|---|---|---|
| `aug` | waveform augmentation before mixup: each op with p=0.3, Gaussian noise std ~ U[0.001, 0.005], circular shift ~ ±U[0.03, 0.08] of the length, gain ~ U[0.9, 1.1] | Sensors 26(10):3214, 2026 (PMC13210826), Table 7 | +1.07 alone over STgram-MFN (92.36 → 93.43); with no mixup in that baseline |
| `gate` | ESTM's TriStat gate: a parameter-free per-frame gate, sigmoid(2·z(median + RMS + var)), applied to the Log-Mel and added as an extra input channel (Mel, EMel, Tgram) | ESTM, arXiv 2509.02471, Table II | +0.86 on its STMamba backbone (94.90 → 95.76), +2.3 on single-path ones |
| `eca` | ECA channel attention after each MobileFaceNet bottleneck's depthwise conv, k=5 | Sensors 26:3214, Table 7 | +0.73 alone over STgram-MFN |

Each gain was measured once, with no seeds, against a weaker baseline. Our
run-to-run noise is 0.19 dev AUC (relu8, n=8). So the prior is that each gain
shrinks a lot on top of OS-SCL. The point is to find which ones survive at n=4.

## Implementation (`ablation/lit-arms` branch, all flags off by default)

| flag | code | notes |
|---|---|---|
| `--wave_aug 1` | `Trainer.augment` | runs on the GPU batch, per clip, before mixup. Touched clips get their mel recomputed on the GPU (~1e-5 dB from the CPU mel); untouched clips keep their cached mel bit for bit (checked). The paper doesn't state the shift direction, so a random sign is used. |
| `--mel_gate tristat` | `model/net.py: tristat_gate`, `input_branches` | **One deviation:** s is z-scored over the clip's frames before alpha=2. ESTM only says "minus μ, the mean used for de-centering". On raw dB the variance term dominates (s has sd 5–34 across frames), and the gate saturates: 78–97% of frames get < 0.1 or > 0.9 on four sample clips. With z-scoring, 25–28% saturate. Computed in fp32. |
| `--eca 1` | `model/net.py: ECA`, `Bottleneck(eca=)` | kernel from ECA's adaptive rule, which gives 5 at both widths (128, 512). +30 parameters. |

**Crash resilience (added 2026-10-17, before any run).** This host's faulty RAM
(`../fast-recipe`) has started crashing processes: the kernel logged 5 Python
general-protection faults or segfaults in one evening, and none before in 49
days of uptime. So every run here passes `--state_every 10`, which overwrites
`state_last.pth` atomically with weights, EMA, optimiser, scheduler, scaler,
best-so-far validation loss and RNG streams. `run_one.sh` retries a crashed
run up to 3 times with `--crash_resume`. The resumed run keeps its own best
validation loss, so checkpoint selection is unchanged. Crashes are logged to
`scores/<run>/crashes.csv`. Tested twice. A 3-epoch run killed mid-epoch-2
resumed at epoch 2 and kept epoch 0 as its best. Through the shared runner
`../train_one.sh`, a `kill -9` (rc 137) mid-epoch-1 retried from the snapshot
and finished with the `done` marker. The runner (GPU lock, idle check, retry,
`wall.csv`, `crashes.csv`) lives at `ablation/train_one.sh`, shared with
`../trunk-soup`. The same code adds
`--state_snapshots` / `--resume_state --fork_seed` for `../trunk-soup`'s forks.

Default models keep identical state_dict keys (checked against
`ablation/fast-recipe`). `eval.py` reads `mel_gate` and `eca` from the
checkpoint's `config.json`. `../ensemble-backend/extract.py` now calls
`net.branch_input` instead of re-stacking the branches itself. That is
identical for existing models, and required for the extra gate channel.

Noise check for `aug`: the median clip RMS is 0.011 for fan, pump, slider and
valve, 0.023 for ToyCar and 0.080 for ToyConveyor. So the paper's noise means
an SNR of 7–21 dB on the MIMII machines and 13–38 dB on the toys.

**Smoke check (2026-10-17):** one 2-epoch run with all three flags on
(`la_smoke`, fussion 4) trains, probes and reloads strictly in `eval.py`.
`extract.forward` matches `net.forward` exactly on that checkpoint. The
cached-dataset path check now compares paths below `data/dataset`: the old
absolute-path check rejected the shared cache from any other worktree.

## Design

- **Base:** the fast recipe (`--amp fp16 --cudnn_bench 1 --lean 1 --cache_dir
  data/cache`, `--num_workers 8`, 300 epochs, `--m 0.4 --ht basic`), with
  `--fussion` as `../fast-recipe` decides (D2: 4 if TFgram stays redundant,
  otherwise 1). `--probe_every 5` stays on; it is logged, never used.
- **Paired baseline:** `../fast-recipe`'s `fr_<base>` runs, seeds 2040–2043, on
  the same host, recipe and code path. The arms use the same seeds. This
  host is not bit-reproducible, so pairing is by seed label only; comparisons
  are of 4-seed means.
- **Arms:** `la_aug`, `la_gate`, `la_eca`, × seeds 2040–2043 = 12 runs.
- **Stage 2 (conditional):** `la_combo` combines every arm that passes D1, at 4
  seeds. If two or more arms pass, it is the candidate single-model system.

Scores per run: eval.py's R0 on dev and eval (the primary score), and S1′
(`../confirm-seeds`' adopted single-model rule, reported).

## Predictions (written before any run)

Δ = arm mean − baseline mean over 4 seeds, R0 overall AUC, in points.

- **P1, `aug`.** Dev Δ in [−0.1, +0.6], central +0.3. ToyCar and ToyConveyor
  gain the most. The 7 dB worst-case SNR may cost pump or valve pAUC: dev pAUC
  on pump falls on ≥ 3/4 seeds.
- **P2, `gate`.** Dev Δ in [−0.2, +0.5], central +0.15. The gate re-weights
  frames, so the gain sits in the non-stationary types: ToyCar and ToyConveyor
  carry ≥ 60% of it.
- **P3, `eca`.** Dev Δ in [−0.2, +0.4], central +0.1.
- **P4.** At most one of the three passes D1.
- **P5.** No arm moves pump's dev AUC by more than ±0.5 on the mean.

## Decision rules

- **D1, an arm passes** if its dev Δ ≥ +0.20 *and* its eval Δ ≥ 0, with both
  4-seed means under R0. The standard error of a 4-vs-4 difference is ~0.13
  (sd 0.19), so +0.20 is about 1.5 se. One arm passing is a lead, not a
  result; `la_combo` must then replicate.
- **D2.** If ≥ 1 arm passes, run `la_combo` (if exactly one passes, it is that
  arm with 4 fresh seeds, 2044–2047, as the replication). Report it against
  the paper under R0 and S1′.
- **D3.** If none passes, drop all three and go to Phase 2 (shared-trunk weight
  soup) on the bare base.

## Addendum before launch (2026-10-23, before any 300-epoch run)

This addendum was written after `../fast-recipe` was scored and before any
lit-arms run trained beyond 2 epochs. It fills in what the pre-registration
left open. It changes no prediction and no decision rule.

- **Base = `fussion 1`** (the paper's TFSTgram), per `../fast-recipe`'s D2.
  That arm cost 0.40 dev AUC without TFgram (`fr_f4` − `fr_f1`), which fails
  D2's −0.15 bar. `run.sh` runs with `LA_FUSSION=1`.
  - **`gate` has four input channels:** Mel, EMel, Tgram and TFgram. That is
    one more than the three-channel (Mel, EMel, Tgram) layout the arm table
    describes for `fussion 4`.
- **Paired baseline** = `fr_f1_s2040–2043` (`../fast-recipe/scores/`), the same
  seeds and recipe. It scores **94.90 dev** (sd 0.34) and **95.25 eval**
  (sd 0.26) AUC under R0.
- **The noise is larger than D1 assumed.** D1's "se ≈ 0.13" came from
  relu8's sd of 0.19. `fr_f1`'s dev sd is 0.34, so a 4-vs-4 difference has
  se ≈ 0.24, and D1's +0.20 bar is only ~0.8 se. D1 stays as registered, but
  every arm is reported with:
  - its Welch se and a bootstrap 95% interval
  - its four seed-paired Δs

  D1 already treats a single pass as "a lead, not a result", and D2's
  fresh-seed replication (2044–2047) carries the evidential weight. An arm
  that passes D1 with an interval that includes 0 is labelled a **weak
  pass**. It still goes to D2.
- **Code.** The branch was fast-forwarded to main (`236cacd`). That brings in
  later flags, all off by default: `--m_warmup`, `--state_snapshots` and the
  mel-free fussion modes. `../margin-warmup` checked that with `--m_warmup`
  off, training is bit-identical to the old code.
  - **Re-smoke on the merged code at `fussion 1`** (2026-10-23): 2 epochs of
    each arm, `la_{aug,gate,eca}_s1_e2`, one per GPU.
  - All three train with 0 skipped steps, write `probe.csv`, and finish
    through `../train_one.sh`.
  - **All three reload strictly in `eval.py` at `fussion 1`.** The smoke
    evals were stopped partway through dev scoring once loading had passed;
    a 2-epoch model's AUC means nothing.
- **Schedule.** 12 runs, arm-major (aug, then gate, then eca; seeds
  2040–2043), three at a time via `run.sh train`.
  - At `fr_f1`'s pace, that is ~11.5 h per run, or ~13.5 h on GPU 1, whose
    PCIe link is x1 (CLAUDE.md). That makes ~2 days in all.
  - Then `LA_FUSSION=1 run.sh score`.
  - S1′ (`../confirm-seeds`) is reported after R0, as registered.
- **Figures.** `make_plots.py` is written during training, before any score
  exists, and reads `scores/` only.

## Results (2026-10-26)

All 12 runs finished 300 epochs with 0 skipped steps. One crashed:
`la_eca_s2041` segfaulted (rc 139) at about epoch 270 and resumed from its
epoch-270 snapshot through `../train_one.sh`; see its `crashes.csv`. Scores
are eval.py's R0 from `scores/<run>/eval_{dev,eval}.csv`. The contrasts come
from `make_plots.py`, which writes `scores/summary.txt`.

### D1: `aug` passes; `gate` and `eca` fail

Δ = arm − `fr_f1`, R0 overall AUC, 4-seed means, in points. Se is Welch's;
the brackets are a bootstrap 95% interval over seeds. Paired Δs are by seed
label (2040, 2041, 2042, 2043).

| arm | dev AUC | dev Δ (se) [95%] | paired dev Δ | eval AUC | eval Δ (se) [95%] | paired eval Δ | D1 |
|---|---|---|---|---|---|---|---|
| `fr_f1` (base) | 94.90 | — | — | 95.25 | — | — | — |
| `aug` | **95.73** | **+0.84** (0.24) [+0.45, +1.25] | +1.02 +0.94 +0.78 +0.61 | 95.45 | +0.20 (0.23) [−0.20, +0.59] | −0.03 +0.33 −0.05 +0.55 | **PASS** |
| `gate` | 94.88 | −0.02 (0.20) [−0.35, +0.30] | +0.41 −0.34 −0.09 −0.06 | 95.65 | +0.41 (0.23) [+0.02, +0.79] | +0.98 −0.28 +0.60 +0.32 | fail |
| `eca` | 94.85 | −0.04 (0.18) [−0.35, +0.26] | +0.23 −0.27 +0.20 −0.34 | 95.77 | +0.52 (0.20) [+0.20, +0.85] | +0.63 +0.13 +0.17 +1.16 | fail |

- **`aug` is a full pass, not a weak one.** Its dev interval excludes 0, all
  four seeds gain (+0.61 to +1.02), and pAUC gains on both sets: dev
  +1.27 [+0.72, +1.86], eval +0.95 [+0.34, +1.63]. The gain shows on the
  whole probe curve from about epoch 150, not only at the end (fig4).
- **But `aug`'s eval AUC gain is small and uncertain.** +0.20, with an
  interval that includes 0. Two of four seeds sit at −0.03 and −0.05. Eval
  ToyCar falls 2.34 points while dev ToyCar rises 1.89. D1 needs eval Δ ≥ 0,
  and +0.20 clears it, but by less than one se.
- **`gate` and `eca` do nothing on dev and gain on eval.** This was not
  predicted and D1 does not act on it: D1 is led by dev. `eca`'s eval gain is
  +0.52 [+0.20, +0.85] with all four seeds positive. Most of it is
  ToyConveyor (+1.69 eca, +2.93 gate), whose seed noise is above 4 points
  (CLAUDE.md). Left as an observation.
- **Cost.** All three add-ons cost little or no training time. On GPU 0,
  `fr_f1` took 39.9k s per run; `aug` took 41.3k s (+3.5%), `gate` 40.4k s and
  `eca` 42.3k s (with its crash and resume). Every difference is within the
  spread between GPUs.

### Per machine (mean Δ AUC vs `fr_f1`, points)

| arm | set | fan | pump | slider | ToyCar | ToyConveyor | valve | overall mAUC |
|---|---|---|---|---|---|---|---|---|
| `aug` | dev | +0.40 | **+1.13** | +0.00 | **+1.89** | **+1.96** | −0.35 | +2.32 |
| `aug` | eval | +0.50 | −0.07 | +0.59 | **−2.34** | +2.57 | −0.06 | −0.20 |
| `gate` | dev | −0.65 | +0.01 | −0.01 | −0.08 | +0.54 | +0.07 | +0.01 |
| `gate` | eval | +0.07 | +0.11 | +0.14 | −0.32 | +2.93 | −0.49 | +0.63 |
| `eca` | dev | +0.21 | −0.06 | −0.03 | +0.48 | −1.06 | +0.21 | −0.14 |
| `eca` | eval | +0.09 | +0.11 | +0.10 | +0.27 | +1.69 | +0.86 | +0.68 |

mAUC is a minimum per machine; deltas under ~2 points are not resolvable
(CLAUDE.md).

### Predictions, checked as written

| | prediction | observed | verdict |
|---|---|---|---|
| P1 | `aug` dev Δ in [−0.1, +0.6] | +0.84 | **fails, too high** |
| | ToyCar and ToyConveyor gain the most | +1.89 / +1.96, the top two | holds |
| | dev pump pAUC falls on ≥ 3/4 seeds | rises on 4/4 (+0.49 to +2.90) | **fails** |
| P2 | `gate` dev Δ in [−0.2, +0.5] | −0.02 | holds |
| | ToyCar + ToyConveyor carry ≥ 60% of the gain | no gain to share | not testable |
| P3 | `eca` dev Δ in [−0.2, +0.4] | −0.04 | holds |
| P4 | at most one arm passes D1 | exactly one | holds |
| P5 | no arm moves dev pump AUC by more than ±0.5 | `aug` +1.13 | **fails** |

The failures are all `aug`, and all in the same direction: it helped more,
and in more places, than the survey's numbers suggested. The worry behind
P1's pump clause, that 7 dB noise would hurt the quiet MIMII machines, was
wrong on dev. Pump gained the most pAUC of any machine after the toys.

### Against the paper

The paper's numbers (`../confirm-seeds`): 95.71 dev AUC, 90.23 dev pAUC,
91.23 dev mAUC, 93.22 eval mean(AUC, pAUC).

- Under R0, `aug`'s 4-seed dev mean is **95.73**, level with the paper's
  95.71. Only one seed clears it (96.20; the others 95.70, 95.54, 95.48).
  `fr_f1` is 0.81 below.
- `aug`'s eval mean(AUC, pAUC) is 93.33, above the paper's 93.22. `fr_f1`'s
  is 92.75.

**Under S1′, as registered.** S1′ is `../confirm-seeds`' adopted
single-model rule: R0 plus a kNN distance, both z-scored with training-part
statistics. `score_s1p.py` computes it from `../ensemble-backend/extract.py`
dumps, using that ablation's rule code unchanged, and writes `scores/s1p.csv`.
`fr_f1` was scored too, as the paired base.

Two checks pass (`scores/s1p_check.json`):

- S0 recomputed from the dumps matches eval.py's committed per-ID AUCs to
  within 2.8e-4.
- The replayed 5% split gives each run's best validation loss to within 7.9e-6.

```
S1′, 4-seed means     dev AUC   dev pAUC   dev mAUC   eval AUC   eval mean(AUC,pAUC)
fr_f1                  95.27     90.01      90.27      95.92      93.41
aug                    96.26     91.31      92.70      96.21      93.96
  Δ vs fr_f1           +0.99     +1.30      +2.42      +0.28      +0.55
gate                   95.30     89.99      90.48      96.20      93.77
eca                    95.26     90.14      90.27      96.29      94.06
paper (one model)      95.71     90.23      91.23        —        93.22
```

- **S1′ keeps D1's verdict.** The dev Δs are `aug` +0.99 (per seed +1.27
  +1.03 +0.87 +0.80), `gate` +0.03 and `eca` −0.01.
- **`aug` with S1′ beats all four of the paper's numbers, on every seed.**
  - dev AUC: 96.07 to 96.52
  - dev pAUC: 90.80 to 91.87
  - dev mAUC: 92.13 to 93.62
  - eval mean(AUC, pAUC): 93.35 to 94.76

  `../confirm-seeds`' single model under S1′ beat only the eval number. Here
  `fr_f1` under S1′ still falls short on all three dev numbers.
- **Not yet a claim.** `aug` was picked as the best of three arms on these same
  dev scores. The replication on fresh seeds (D2, below) decides whether this
  holds.

### Decision

**D2 applies, as the replication branch:** exactly one arm passes, so
`la_combo` is `aug` alone on 4 fresh seeds, 2044–2047, run with
`LA_FUSSION=1 LA_ARMS=aug LA_SEEDS="2044 2045 2046 2047" run.sh train`. D3
(drop all three, go to trunk-soup on the bare base) is not triggered.

Predictions for the replication, written before it trains:

- **R1.** The fresh-seed `aug` dev mean beats `fr_f1`'s (94.90) by ≥ +0.20.
  Central +0.5: `aug` was picked as the best of three, so its +0.84 should
  shrink toward the mean.
- **R2.** Its eval Δ is ≥ 0. Likelier to fail than R1: stage 1's eval gain was
  +0.20 with 2/4 seeds below 0.
- **R3.** Pooled over all 8 `aug` seeds, the dev Δ against `fr_f1`'s 4 has a
  bootstrap 95% interval that excludes 0.

`aug` is adopted into the recipe if R1 and R2 both hold. The baseline stays
`fr_f1_s2040–2043`: this host is not bit-reproducible, so pairing by seed
label carried no information, and training fresh baseline seeds would double
the cost. The comparison is unpaired, 4 against 4.

## Figures

| figure | what it shows |
|---|---|
| `plots/fig1_auc.png` | dev and eval AUC per seed for the base and each arm; bar = mean. `aug` separates on dev, not on eval |
| `plots/fig2_contrasts.png` | each arm's Δ against `fr_f1`: mean, ±1 se, bootstrap 95%, paired Δs, and the D1 bars |
| `plots/fig3_machines.png` | Δ per machine type. `aug`'s dev gain is ToyCar, ToyConveyor and pump; eval ToyCar goes the other way |
| `plots/fig4_probe.png` | dev AUC probe every 5 epochs, arm vs base. Diagnostic only; never used to pick a checkpoint |
