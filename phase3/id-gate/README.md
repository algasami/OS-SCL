# id-gate — can a normal-only proxy tell, per machine ID, where BEATs helps?

> **Post hoc (phase-3 rule 2): no network is trained, and every member's eval scores have been seen.**
> **Stage 1 is a diagnostic that fits nothing and reads dev labels only.** If it fails, the study
> ends there. Stage 2, an actual gate, gets its own pre-registration, committed before its eval is scored.

## Question

The suggestion came from the user's senior: a mixture-of-experts "embedding gate" that decides per
input how much to trust each expert. `../beats-router` showed why a gate fitted on dev labels fails
here. Dev and eval test sets hold **disjoint machine IDs**, and BEATs' effect per machine type flipped
sign between them on 3 of 6 types.

A gate can avoid that only if its signal is computed **on the target IDs themselves, without anomaly
labels**. Every ID, eval ones included, has normal training clips. The proxy:

> For ID i, score the training clips of ID i (normal) and the training clips of the *other* IDs of
> the same machine type (stand-in anomalies) as if they belonged to ID i. Then measure how well each
> system separates them.

If the proxy's per-ID gain from adding BEATs tracks the true per-ID gain on dev, a per-ID BEATs weight
could be set on eval IDs from their training clips alone.

## Stage 1 setup (dev only, fits nothing)

- **Systems.** The same members and block weights as X1f vs GS12f:
  - from-scratch block: 8 `fc_nogain` (2060–67) and 4 `fc_spec` (2060–63), weight 1 each;
  - BEATs block: `bt_arc_s2040–42`, weight 1.5 each.
  - Dumps come from `../OS-SCL-syn-fresh` and `../OS-SCL-beats` via a symlink tree. The machinery
    (`ablation/ensemble-backend/score.py`: `l2`, `nn_dist`, `z`) is imported, not rewritten.
- **Proxy score of a clip x for ID i, per member.** zKp: the 1-NN cosine distance from x's
  embedding (`pre`, 128-d) to ID i's reference bank (its training-part clips, `~valid`), z-scored with
  ID i's leave-one-out reference distances. This is exactly the kNN half of S1′, and all of zK′ for BEATs.
  - **Only the kNN half can be used.** The CE half of S1′ needs logits for ID i on another ID's clip,
    and the dumps store CE against a clip's own label only. The proxy for the from-scratch block is
    therefore kNN-only, while its test score is CE + kNN. This is a known mismatch.
- **Clips.**
  - Positives (normal): all training clips of ID i. A clip in ID i's bank is scored leave-one-out,
    excluding itself.
  - Negatives (stand-in anomalies): all training clips of the other IDs of the same machine type,
    scored against ID i's bank.
  - Training clips are used rather than the 5 % held-out split, because each member's held-out split
    depends on its seed. A clip held out by one member sits in another member's bank. Leave-one-out
    gives every member the same clips.
- **Proxy per ID.** For each system s ∈ {GS12f-kNN, X1f-kNN}, the per-clip score is Σ w_member · zKp,
  and proxyAUC_i(s) is the ROC AUC of negatives vs positives. pAUC (max FPR 0.1) is computed beside it.
  - **π_i** = proxyAUC_i(X1f-kNN) − proxyAUC_i(GS12f-kNN). Primary; it mirrors the decision the gate would make.
  - π^b_i = proxyAUC_i(BEATs block) − proxyAUC_i(from-scratch block). Secondary: block against block.
- **Truth per ID (dev only).** Δ_i = Amean_i(X1f) − Amean_i(GS12f), where Amean_i is the mean of AUC
  and pAUC on ID i's dev test clips (`eval.py`'s per-ID metrics, via `ge.evaluator(...)(…, per_id=True)`).
  There are 23 dev IDs.
- **Eval.** π_i is computed for the 18 eval IDs (training clips only, no labels) and written to
  `scores/`. **Eval Δ_i is not computed in stage 1.**

### Decision rule D0 (stage 1)

Stage 2 is opened only if all three hold on the 23 dev IDs:

- (a) **not saturated**: fewer than half the IDs have proxyAUC ≥ 0.999 under *both* systems. If they
  are saturated, π_i carries no information;
- (b) Spearman ρ(π_i, Δ_i) ≥ 0.40;
- (c) the 95 % bootstrap CI of ρ, resampling IDs with 2000 draws and seed 0, excludes 0 from above.

Descriptive only, never a decision:
- ρ within machine type (π and Δ centred per type);
- ρ for π^b;
- the pAUC versions;
- a scatter plot of π_i against Δ_i.

**If D0 passes**, a stage-2 pre-registration is committed before any eval Δ_i is computed. It fixes the
gate form (per-ID BEATs weight as a function of π_i), chosen on dev only, plus its decision rules.
Eval is then scored once. **If D0 fails**, the study ends. The embedding-gate idea is then recorded as
"no normal-only signal found".

## Predictions (written before anything is computed; 80 % intervals)

| ID | quantity | prediction |
|---|---|---|
| G1 | median over dev IDs of proxyAUC(GS12f-kNN) | 0.995 [0.97, 1.000]. Other IDs are what the networks were trained to tell apart |
| G2 | D0(a) saturated (fails) | 45 % |
| G3 | Spearman ρ(π, Δ), dev | +0.10 [−0.30, +0.45] |
| G4 | D0 overall | **fails** (80 %) |
| G5 | within-type ρ | +0.05 [−0.35, +0.40] |

## Results

*(not yet computed)*
