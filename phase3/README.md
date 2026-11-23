# phase3/ — studies after synthesis

`ablation/` (phase 1, 26 studies, closed 2026-11-08) asked one question per study.
`synthesis/` (phase 2, closed 2026-11-12) combined the adopted findings and confirmed them on fresh
seeds. **Every study started after 2026-11-12 lives here**, one directory per study, with the same
layout as the earlier phases (`README.md`, `run.sh`, `make_plots.py`, `plots/`, `scores/`,
gitignored `results/`).

## Where phase 2 ended (the incumbents)

Amean = mean(AUC, pAUC), each pooled per machine as `eval.py` does, averaged over the 6 machines.
Paper: dev AUC / pAUC / mAUC 95.71 / 90.23 / 91.23, eval mean 93.22.

| track | system | dev Amean | eval Amean | source |
|---|---|---|---|---|
| single network | `ao_nogain` under C1ᵀ, 8 fresh seeds (2060–67) | 94.11 | 94.37 | `synthesis/ensemble-g8` |
| from-scratch ensemble | **GS12f** = 8 `ao_nogain` + 4 `sb_spec`, equal weights (reported) | 95.01 | 95.03 | `synthesis/ensemble-g8` |
| from-scratch ensemble | G8f = 8 `ao_nogain` (recipe for a new build) | 94.76 | 95.00 | `synthesis/ensemble-g8` |
| with external pretraining | **X1f** = GS12f + 3 BEATs × 1.5 | 95.19 | 95.51 | `synthesis/ensemble-g8` |

## Studies

| study | question | compute | branch / worktree |
|---|---|---|---|
| `beats-router/` | Can X1f run its BEATs block only on some machine types, chosen on dev, and keep its accuracy at a fraction of the inference cost? **Done 2026-11-22: no.** The dev pick R\* saves 42 % of FLOPs but loses 0.11 [−0.22, −0.03] eval Amean to X1f; dev and eval disagree on where BEATs helps. X1f stays. | CPU (existing dumps) + a short GPU timing | `phase3/beats-router` / `../OS-SCL-p3-router` |

## Rules of this phase (not optional)

1. **Everything in phase 2's rules still holds** (`synthesis/README.md`, *Rules*): dev-only
   selection, rules and predictions committed before scoring, eval scored once, the phase-1 noise
   floors, external pretraining as a separate track, layout and commit format per CLAUDE.md.
2. **Every existing network's eval scores have been seen**, including the fresh seeds 2060–67.
   A study that only re-combines existing networks is **post hoc** and says so in its first lines.
   A clean claim needs networks trained for it, under a pre-registration written before training.
3. **Say what is arithmetic before predicting it.** If a quantity already follows from committed
   numbers, state it as such; predictions are reserved for quantities that are genuinely unknown.
4. Ensemble deltas under ~0.15 Amean are seed noise (`synthesis/ensemble-g8`); don't read them.
5. Studies run on their own branch (`phase3/<study>`) in their own worktree, branched from
   `phase3/base`. Merging `phase3/base` into `main` waits for the user.
