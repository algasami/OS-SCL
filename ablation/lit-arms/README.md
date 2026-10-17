# Literature arms: three published add-ons, each at n=4 on the fast recipe

> **Status: pre-registered 2026-10-17. Nothing trained.** Phase 1 of the plan.
> Waits for `../fast-recipe` to fix the base model (its decision D2) and to free
> the GPUs.

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

## Results

*(empty until the runs finish)*

## Figures

*(empty until the runs finish)*
