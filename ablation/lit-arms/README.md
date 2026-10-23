# Literature arms: three published add-ons, each at n=4 on the fast recipe

> **Status: training since 2026-10-23, base `fussion 1`.** Pre-registered
> 2026-10-17. A pre-launch addendum (2026-10-23, below Decision rules) fixes
> the base and records the measured noise. The predictions and rules are
> unchanged.

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

## Results

*(empty until the runs finish)*

## Figures

*(empty until the runs finish)*
