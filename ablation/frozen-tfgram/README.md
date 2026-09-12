# Frozen TFgram — is the branch *learned*, or a fixed random projection?

Follow-up to [`../gram-branch-redundancy`](../gram-branch-redundancy), which
found TFgram largely redundant with Log-Mel (CKA 0.89, R² 0.63) and showed that
dropping it entirely (f4) matches the full model. That leaves one thing
unresolved: f4 also *removes a channel*, so it cannot separate

- "TFgram's **learned** features contribute nothing", from
- "TFgram contributes as a **fixed random projection**, and training it adds nothing".

This ablation separates them. Both arms keep `fussion=1` — all three input
channels, unchanged architecture, unchanged shapes — and only stop the TFgram
branch from learning. Any gap vs f1 is then attributable to TFgram's *learning*
alone, with the channel-count confound held fixed.

## Arms

`--freeze_tfgram {strict,bn_adapt}` (new; default `none` = unchanged behaviour).

| arm | TFgram weights | TFgram BatchNorm | branch output std |
|-----|----------------|------------------|-------------------|
| `ft_strict` | frozen at random init | pinned to `eval()`; running stats stay at init, so BN is an identity | 0.024 |
| `ft_bnadapt` | frozen at random init | still tracks the data, keeping the random features sanely scaled | 0.668 |

Both variants are run because the BN treatment is not a detail: a strict freeze
leaves the branch 27x weaker in scale, so on its own it would confound "a random
projection is enough" with "the channel is effectively dead". `bn_adapt` is the
classic random-features control; `strict` is the literal one.

## Why the trainable-parameter count makes this clean

```
f1  (mel+Tgram+TFgram)    1,377,535 used     1,377,535 trainable
f4  (mel+Tgram)           1,178,239 used     1,178,239 trainable
frozen arms               1,377,535 used     1,178,815 trainable
```

The frozen arms have the same trainable capacity as f4 to within **576
parameters** — exactly the one extra `MobileFaceNet.conv1` input channel — while
still receiving all three channels. So the triangle f1 / f4 / frozen isolates the
variable cleanly:

- **frozen ≈ f1** → TFgram acts as a fixed random projection; training it buys
  nothing, and the paper's contribution does not survive the control.
- **frozen ≈ f4** → the frozen channel is inert; consistent with TFgram being
  redundant with Log-Mel either way.
- **frozen < both** → a randomly-initialised TFgram actively injects noise, i.e.
  the branch has to be trained to avoid doing harm.

## The trap this ablation had to work around

`requires_grad = False` is **not** sufficient to freeze a branch in this repo.
`WeightEMA.step()` (`losses.py`) iterates the whole `state_dict` and applies
`param.mul_(1 - wd)` with `wd = 2e-6` on every step, outside the optimizer. At
539 steps/epoch x 300 epochs = 161,700 steps that is a decay to **0.72x**, and it
hits BatchNorm buffers as well as weights. So `freeze_tfgram()` returns the
state_dict keys to exclude and `WeightEMA` takes a `skip_keys` argument. Passing
no `skip_keys` reproduces the original behaviour exactly, and the pre-existing f1
checkpoint still loads with `strict=True`.

`strict` mode has a second hazard: `trainer.py` calls `net.train()` once per
iteration, which would put the branch back in train mode. `nn.Module.train()`
recurses into children, so `freeze_tfgram` stubs out `TFgramNet.train` to stop
the recursion there.

Both were caught by a pre-flight check that runs real training steps and asserts
the branch is bit-identical afterwards, in the student *and* the EMA net.

## Running it

```bash
bash ablation/frozen-tfgram/run.sh          # both arms, GPUs 0 and 1, ~5 h
python eval.py --m 0.4 --fussion 1 --ht basic --gpu_num 0 \
    --model_path check_points/SCLTFSTgramMFN_ft_strict/model.pth --d --csv
```

Arms are `fussion=1`, so eval needs no special flags. Seed 2024, matching f3/f4.

## Results

Single seed (2024) per arm, 300 epochs, both arms concurrent on GPUs 0/1.
Baseline `f1` and the `f3`/`f4` rows are carried over from
[`../gram-branch-redundancy`](../gram-branch-redundancy).

| arm | TFgram | trainable | wall clock | best epoch | dev AUC/pAUC/mAUC | eval AUC/pAUC/mAUC |
|-----|--------|-----------|-----------|-----------|-------------------|--------------------|
| f1 | trained | 1,377,535 | — | — | 94.95 / 89.67 / 89.22 | 94.84 / 90.51 / 90.48 |
| f3 | trained, no Tgram | 1,096,553 | 5:24 | — | 94.30 / 88.77 / 87.39 | 95.99 / 91.11 / 92.00 |
| f4 | removed | 1,178,239 | 3:10 | — | 95.01 / 89.57 / 89.17 | 95.64 / 91.49 / 91.63 |
| `ft_bnadapt` | frozen, BN adapts | 1,178,815 | 4:05 | 280 | 94.92 / 89.53 / 89.17 | 95.59 / 90.94 / 92.08 |
| `ft_strict` | frozen, BN pinned | 1,178,815 | 4:19 | 281 | **95.36 / 89.62 / 90.61** | **95.98 / 91.23 / 92.26** |

Deltas vs f1 (percentage points):

```
                    dev                       eval
f3           -0.64  -0.90  -1.83      +1.15  +0.60  +1.52
f4           +0.06  -0.10  -0.05      +0.80  +0.98  +1.15
ft_bnadapt   -0.02  -0.14  -0.05      +0.75  +0.43  +1.61
ft_strict    +0.41  -0.05  +1.39      +1.14  +0.72  +1.78
```

## Conclusion

**Training TFgram buys nothing.** Neither frozen arm loses to f1 on either test
set. A TFgram branch left at its random initialisation — never updated, in
`strict` mode never even renormalised — performs as well as one trained for 300
epochs, with the channel count and every shape held fixed. Whatever the branch
contributes, it contributes as a *fixed random projection*.

This is a stronger statement than the parent ablation could make. `f4` showed the
branch is **removable**; these arms show its **learning** is inert while the
branch is still there, which is the version of the claim that survives the
"you also deleted a channel" objection.

Both frozen arms land on `f4`, not between `f4` and `f1` (dev: `ft_bnadapt` is
0.09 below `f4`, `ft_strict` 0.35 above; eval: 0.05 below and 0.34 above). So
freezing costs about what deleting costs — i.e. nothing.

### The gains are ToyConveyor, and should not be claimed

Every arm here "beats" f1 on the evaluation set, and it is all one machine.
Per-machine AUC delta vs f1:

```
dev             fan   pump slider ToyCar  ToyConv  valve | mean excl. ToyConv
f3            -2.89  -1.09  +0.06  +0.16   -0.14  +0.04  |  -0.74
f4            -0.03  -0.01  -0.03  +0.06   +0.46  -0.08  |  -0.02
ft_bnadapt    +0.08  -0.05  -0.00  -0.16   +0.22  -0.21  |  -0.07
ft_strict     +0.38  -0.20  -0.02  +0.07   +2.31  -0.09  |  +0.03

eval            fan   pump slider ToyCar  ToyConv  valve | mean excl. ToyConv
f3            +0.10  +0.05  -0.45  +0.50   +6.14  +0.58  |  +0.16
f4            +0.20  -0.13  -0.07  -0.43   +4.91  +0.30  |  -0.02
ft_bnadapt    +0.15  -0.42  -0.18  +0.12   +4.36  +0.47  |  +0.03
ft_strict     +0.01  -0.36  -0.04  -0.13   +6.91  +0.47  |  -0.01
```

Drop ToyConveyor and the whole table collapses to a null result: every TFgram
variant — trained, frozen, or deleted — is within **0.07 AUC on dev and 0.03 on
eval**. Only `f3`, which removes *Tgram*, moves (-0.74 dev), and it moves through
fan (-2.89) and pump (-1.09).

f1's ToyConveyor eval score (78.28) is simply a bad draw; every other arm lands
82-85. The parent README already flagged ToyConveyor as this reproduction's
unstable machine, and a +4.4 to +6.9 point spread on one machine across five runs
is what that instability looks like. So the defensible claim is
**"freezing TFgram costs nothing"**, not "freezing TFgram helps".

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/frozen-tfgram/make_plots.py`
(matplotlib lives in the `base` conda env, not `os-scl`). All four read from the
eval CSVs and the training logs — nothing is hardcoded.

Every figure is *relative*: f1 is the zero line rather than a plotted series,
which also keeps each categorical figure to three colour slots — the cap that
validates under `--pairs all` (CVD ΔE 9.2, normal-vision ΔE 24.0). `f4` keeps the
aqua it has in `../gram-branch-redundancy` so the two figure sets agree.

| figure | shows |
|--------|-------|
| `plots/fig1_toyconveyor_collapse.png` | dumbbell: mean per-machine AUC delta vs f1, all six machines vs ToyConveyor dropped — every apparent gain collapses to ≈0 |
| `plots/fig2_per_machine_delta.png` | diverging matrix of per-machine AUC delta vs f1, all four arms × both sets; the ToyConveyor column is the whole story |
| `plots/fig3_null_zoomed.png` | the claim at high zoom: ToyConveyor excluded, ±0.7 axis, trained/frozen/removed indistinguishable on every machine |
| `plots/fig4_training_curves.png` | validation loss and the accuracy plateau — freezing the branch does not impair optimisation |

## Caveats

Single seed per arm, as everywhere in this directory. The parent README's warning
holds: ~1-point differences are not resolvable at n=1, and the ToyConveyor spread
above is the direct evidence for that. What makes this more than noise is not the
magnitude but the agreement — the frozen arms match f1 on dev *and* eval *and*
per-machine outside ToyConveyor, and the CKA probe independently predicted it
(TFgram at 0.89 CKA with Log-Mel; a branch that mostly re-derives its neighbour
has little left to learn).

Still missing, in priority order: a 3-5 seed sweep (`--seed`) over f1/f4/frozen,
and a local `--fussion 2` Log-Mel-only run to anchor the low end.
