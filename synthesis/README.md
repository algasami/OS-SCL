# synthesis/ — phase 2: combine what the ablations found into the best system

`ablation/` (phase 1, 26 studies, closed 2026-11-08) asked one question per study.
This phase asks no new questions about the model. It **combines adopted findings**,
picks final systems by a rule fixed in advance, and checks them on seeds nobody has seen.

> **Status 2026-11-11: closed.** All four studies are done; the answer is in `final/README.md`.
> One `ao_nogain` network scored by C1ᵀ holds on fresh seeds (Amean 94.02 dev / 94.23 eval, paper
> 92.97 / 93.22). A from-scratch ensemble reaches about 94.9 on eval; `sb_spec` members help on
> dev and are not shown to help on eval. With BEATs (separate track) X1 reaches 95.42. The text below is the charter as
> written on 2026-11-09 and is left unchanged.
>
> **Follow-up 2026-11-12 (`ensemble-g8`).** Four more fresh `ao_nogain` networks, rules committed
> before scoring. One network under C1ᵀ: 94.11 dev / 94.37 eval over 8 fresh seeds, each above every
> paper number (under `eval.py`'s own score, 4 of 8). Reported from-scratch ensemble: GS12f
> (8 `ao_nogain` + 4 `sb_spec`), **95.01 / 95.03**; 8 `ao_nogain` networks alone reach 94.76 / 95.00,
> so `sb_spec` is still not shown to help on eval. With BEATs: X1f, 95.19 / 95.51. `final/README.md` carries these as **[g8]** notes.

## Where phase 1 ended (the incumbents)

Amean = mean(AUC, pAUC), each pooled per machine as `eval.py` does, then averaged
over the 6 machines. The paper reports dev AUC / pAUC / mAUC 95.71 / 90.23 / 91.23
and an eval mean of 93.22.

| track | system | dev AUC / pAUC / mAUC | dev Amean | eval Amean | source |
|---|---|---|---|---|---|
| single network | `ao_nogain` under C1, 8 seeds | 96.55 / 91.78 / 93.22 | 94.16 | 94.09 | `ablation/pooled-knn` |
| single network | `ao_nogain` under R0 (eval.py), 8 seeds | 96.10 / 91.08 / 92.12 | — | 93.14 | `ablation/aug-ops` |
| ensemble | E2′[A8] (8 `la_aug`) — reported best | 97.05 / 92.48 / 94.10 | 94.77 | 94.78 | `ablation/mixed-ensemble` |
| ensemble | E2′[G8] (8 `ao_nogain`) — dev pick, tied A8 on eval | 97.19 / 92.76 / 94.12 | 94.97 | 94.87 | `ablation/nogain-ensemble` |
| external pretraining | E2′[A8] + ΣS1′[B3] (F1) | — | 94.93 | 95.15 | `ablation/beats` |
| external pretraining | E2′[A8] + ΣzK′[B3] (F1k, CE-free, secondary) | — | 95.01 | 95.25 | `ablation/beats` |

Adopted findings that have **never been combined**:
- **C1** (S1′ + zGmean, `pooled-knn`) and **small TTA** (shifts 0, ±1 %, `tta`) are both
  adopted single-model rules, each tested only without the other.
- The ensembles were only ever built from **one member type at a time** (or the
  fixed mixes in `mixed-ensemble`). The pool now holds ~60 dumped from-scratch members of
  8 kinds: `ao_nogain` ×8, `la_aug` ×8, `fr_f1`/`la_gate`/`la_eca` ×4, `sb_spec` ×4,
  `tp` ×4, trunk-soup forks ×8, `pm` ×4. Several failed as single models but err
  differently (spectrum-branch: pump id_02 +2.9; timepool-head: better embedding kNN).
- BEATs (external pretraining) helps only through fusion, mainly eval ToyCar.

Findings that **failed** and are not revisited: PerMix loss, time-pooled `linear7`,
spectrum branch as a single model, trunk soup, distillation, max-pooled/sub-band kNN,
patch kNN, `match` TTA, margin warm-up on the full model.

## Studies

| study | question | compute | branch / worktree |
|---|---|---|---|
| `single-final/` | Does C1 + TTA (C1ᵀ) beat C1 for one `ao_nogain` network? | CPU (existing dumps) | `synthesis/single-final` / `../OS-SCL-syn-single` |
| `grand-ensemble/` | Which ensemble of the existing pool is best, picked on dev only? With and without BEATs. | CPU (existing dumps) | `synthesis/grand-ensemble` / `../OS-SCL-syn-ens` |
| `fresh-confirm/` | Do the frozen final rules hold on seeds trained now (2060+), never seen under any rule? | GPU (training) | `synthesis/fresh-confirm` / `../OS-SCL-syn-fresh` |
| `final/` | The final systems, one table, one script, figures, independent review. | CPU | `synthesis/base` |

## Rules of this phase (not optional)

1. **Every existing member's eval scores have been seen**, under R0, S1′ and usually C1.
   A new *combination* is therefore selected on **dev only**. Its candidate list, the
   selection procedure and the decision rule are committed **before** that combination is
   scored on eval, and eval is scored once.
2. **Fresh seeds (2060+) are untouched.** They are not scored on dev or eval under any
   rule until `single-final` and `grand-ensemble` have committed their final rules. The
   training probe (`--probe_every 5`) is kept for recipe identity; its log is not read.
3. **Predictions before results**, as in phase 1 — README question, arms, numeric
   predictions and decision rules are committed first.
4. **Noise floors from phase 1 still apply:** ~1 point eval AUC run-to-run, mAUC deltas
   under ~2 points unresolvable, ToyConveyor >4 points. Report paired seed deltas with a
   bootstrap CI, never a single-seed difference.
5. **External pretraining is a separate track.** Every BEATs-containing number carries the
   label "with external pretraining" and never enters a from-scratch table.
6. Layout, `scores/` vs `results/`, curve distillation, figure palette and commit format are
   exactly `ablation/`'s (see CLAUDE.md). `ablation/scores_path.py` also resolves
   `synthesis/*/scores/<model_name>`; `ablation/train_one.sh` takes `ABL_ROOT=synthesis`.
7. Merging to `main` waits for the user.
