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

> **Verdict: D0 fails on (a). The proxy is fully saturated, so the study ends at stage 1.** Every one
> of the 41 IDs (23 dev, 18 eval) gets proxyAUC = 1.0 and proxy pAUC = 1.0 under every system. Other
> IDs' clips are separated perfectly from ID i's own, because telling IDs apart is exactly the task the
> networks were trained on. So π_i = 0 on every ID up to float roundoff (|π| ≤ 1.1e-16), and it cannot
> say where BEATs helps. **The embedding-gate idea is recorded as "no normal-only signal found" with
> this proxy.**

`run.sh stage1` was replicated byte for byte. No eval-set metric of any system was computed.

**Checks.**
- The proxy's zKp construction reproduces the adopted test-clip zKp for all 15 members: max |diff| is
  1.4e-13 for the from-scratch members and 2.1e-11 for BEATs.
- X1f and GS12f reproduce the committed dev Amean to ≤ 4.4e-5.
- **Per-ID pAUC.** `ge.evaluator(..., per_id=True)` returns per-ID AUC only. Per-ID pAUC was therefore
  computed with `eval.py`'s own sklearn call (`max_fpr=0.1`) on each ID's clips. Averaged back up, the
  per-ID numbers match `fc.metrics` to 1e-14.

| D0 part | needs | got | |
|---|---|---|---|
| (a) not saturated | < 11.5 of 23 IDs at proxyAUC ≥ 0.999 under both systems | **23 of 23**. Smallest gap from lowest stand-in anomaly to highest normal: 209 summed-z units (from-scratch), > 1400 (X1f-kNN) | **fail** |
| (b) ρ(π, Δ) ≥ 0.40 | | −0.25 | fail |
| (c) bootstrap CI > 0 | | [−0.48, −0.06]; 238 of 2000 draws undefined | fail |

- **The ρ values are not a signal.** They are computed on roundoff: 21 of 23 π values are exactly 0,
  and there are only 3 distinct values. The same holds for within-type ρ (−0.24) and ρ(π^b, Δ) (−0.13).
  pAUC-based ρ is undefined, since π^pAUC is identical on every ID. D0(a) is the part of the rule built
  to catch this.
- **The truth side does vary.** Per-ID dev Δ_i runs from −0.71 (ToyConveyor-03) to +2.26 (pump-02),
  and 9 IDs sit at 0. Per ID, there is something to gate on. The proxy just cannot see it.

| ID | predicted | got | |
|---|---|---|---|
| G1 | median proxyAUC 0.995 [0.97, 1.000] | 1.000 | hit |
| G2 | saturated, 45 % | saturated, 23/23 | happened, though I put it under even odds |
| G3 | ρ +0.10 [−0.30, +0.45] | −0.25 | inside the interval, but vacuous (roundoff) |
| G4 | D0 fails (80 %) | fails | hit |
| G5 | within-type ρ +0.05 [−0.35, +0.40] | −0.24 | inside the interval, but vacuous (roundoff) |

## What this means for the embedding gate

- **Other IDs are useless as stand-in anomalies.** The networks are ID classifiers, so other IDs are
  the easiest possible "anomalies". Every system, with or without BEATs, separates them perfectly.
- **The other routes are closed or unpromising:**
  - a gate fit on dev labels inherits the dev → eval ID shift (`../beats-router`);
  - a learned clip-level gate has no anomaly labels to train on.
- **One door stays open, and it is not tested here: harder stand-in anomalies.** Synthetic perturbations
  of ID i's own clips (noise, pitch or time shifts, mixing in another machine at low SNR) might not
  saturate. That would be a new study with its own pre-registration and a generation step. Its prior is
  low: synthetic anomalies are a known weak proxy for real faults in DCASE 2020.
- **Practical conclusion for the senior's suggestion.** In this setup, the useful MoE-style change was
  shrinking the BEATs block (`../beats-count`, X1f-lite), not gating it.

## Figures

| file | shows |
|---|---|
| `plots/fig1_pi_vs_delta.png` | π_i against Δ_i for the 23 dev IDs: every π sits at 0 while Δ spans −0.71 to +2.26 |
| `plots/fig2_proxyauc_saturation.png` | proxyAUC per system and machine type, dev and eval IDs: every point at 1.0, above the 0.999 line |

## Reproduce

```bash
bash phase3/id-gate/run.sh links
bash phase3/id-gate/run.sh stage1      # replicated until two tries agree byte for byte -> scores/
/home/f74134118/anaconda3/bin/python phase3/id-gate/make_plots.py
```
