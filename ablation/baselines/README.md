# Shared baselines

Runs that no single ablation owns, kept here so every ablation can read them
through `scores_path.scores_dir()` without relative paths.

| run | what it is |
|-----|------------|
| `SCLTFSTgramMFN_test` | this machine's reproduction of the paper's TFSTgram (`fussion=1`), the `f1` baseline in all three ablations |
| `SCLTFSTgramMFN` | a default-named run; config only, no scores |

## `SCLTFSTgramMFN_test` is not reproducible from any `run.sh`

It predates the ablation directories. To recreate it:

```bash
python train.py --m 0.4 --gpu_num 0 --fussion 1 --ht basic --desc test
```

**It must run with `--num_workers 0`** (the current default — do not pass `8`).
`persistent_workers` changes the torch RNG stream from epoch 1, so the same seed
under `--num_workers 8` produces a *different* run: this baseline scores 94.84
eval AUC at `nw=0` and 95.85 at `nw=8`. That difference silently inflated the
eval-side results of the first two ablations; see
[`../seed-sweep`](../seed-sweep).

Its training log was never committed (`*.log` is gitignored), so `curve.csv` is
absent and cannot be recovered.
