# HANDOFF — synthesis phase (written 2026-11-10 17:55 CST)

> Working note for the next session, not a study. **Delete it before `synthesis/base` merges into
> `main`.** The charter is `synthesis/README.md`. Durable state also lives in the auto-memory
> `project-synthesis-phase.md`.

## The task (user, 2026-11-09)

Orchestrate research and experiments to build the best model from all ablations done so far.
- Put the work in `synthesis/`, not `ablation/`, as a clear phase change.
- Combining existing ablations is preferred over running new ones.
- Subagents work on separate worktrees and branches.

Standing rules (CLAUDE.md + memory):
- Commit messages are one line of at most 120 characters, with no attribution or session trailers.
- Run long jobs in the background and only interrupt the user for real decisions.
- Every study ships README predictions, a `scores/` record and plots.
- Nothing is tracked under `results/`.
- **Ask the user before merging to `main`.**

## Where things stand

| study | branch / worktree | state |
|---|---|---|
| `single-final` | `synthesis/single-final` / `../OS-SCL-syn-single` | DONE, merged into base. **C1ᵀ adopted:** dev 94.26, eval 94.10. |
| `grand-ensemble` | `synthesis/grand-ensemble` / `../OS-SCL-syn-ens` | DONE, merged into base. **GS12** dev pick ties A8 (eval 94.93 vs 94.78, CI spans 0), so A8 stays the reported best. **X1** = GS12 + 1.5·BEATs zK′, eval 95.42, +0.17 [+0.02, +0.32] vs F1k: adopted, thin. |
| `fresh-confirm` | `synthesis/fresh-confirm` / `../OS-SCL-syn-fresh` | **TRAINING, then auto-extract.** Rules + predictions F1–F6 frozen in d3472f2; scorer selftest passed (5c41150); `post` stage 9779ed4. |
| `final/` | `synthesis/base` / `../OS-SCL-syn` | **DRAFT** daa6a4a: `collect.py` → `scores/summary.csv`, `make_plots.py` fig1/fig2; fig3 and fresh rows appear once fresh-confirm is scored. |

`main` (`/home/f74134118/OS-SCL`) is untouched since db5e457.

## What is running (all detached: own session, no tty, PPID 1 — survives a session reset)

| process | what it does | expected end |
|---|---|---|
| `bash synthesis/fresh-confirm/run.sh train` (pid 12740, in `../OS-SCL-syn-fresh`) | Round 2 `fc_spec_s2060–2063`. s2061 is on GPU 1 (started 06:15), s2060 on GPU 0 (15:12), s2063 on GPU 2 (15:56). s2062 is already done. | ~20:30, ~02:45, ~03:15 (11-11) |
| `bash synthesis/fresh-confirm/run.sh post` (pid 10641) | Waits for every `train.py --desc fc_*` and the round-2 runner to exit. Then checks 8/8 `done`, runs `records`, then `extract` (3 tries) on GPU 0. **Stops before `score`.** | ~1–2 h after training |

Round 1 `fc_nogain_s2060–2063` finished 11-10 03:49–15:08 with no crashes.

### Check progress

```bash
cd /home/f74134118/OS-SCL-syn-fresh
cat synthesis/fresh-confirm/results/post.status        # WAITING | RECORDS | EXTRACT try n | READY_TO_SCORE | FAIL ...
grep -h "TRAIN\|FAIL\|RETRY\|train end" synthesis/fresh-confirm/results/run_train*.log
ls check_points/SCLTFSTgramMFN_fc_*/done | wc -l      # 8 when training is complete
nvidia-smi
```

**Do not read** the fresh runs' probe values (the `probe |` lines in `results/train_fc_*.log`, or
`probe.csv`). They are dev-test scores, and the charter keeps fresh seeds unscored until the
frozen rules run.

## Next steps, in order

1. **Wait for `post.status` = `READY_TO_SCORE`.**
   - If it says `FAIL only n/8`, see *Recovery*.
   - If it says `FAIL extract`, rerun `FC_GPU=0 bash synthesis/fresh-confirm/run.sh extract`. It resumes: existing `<run>.npz` files are skipped.
2. **Score ONCE:** `cd ../OS-SCL-syn-fresh && bash synthesis/fresh-confirm/run.sh score`.
   - Runs on CPU; every computation is done twice because of the faulty RAM.
   - Writes `scores/single.csv`, `single_machines.csv`, `ensembles.csv` (G4, GS8, X1), `bootstrap.json` and `decisions.json` (F1–F6).
   - If it crashes partway (segfault = RAM), rerun it. It is deterministic, and nothing counts until it completes.
3. `FC_GPU=0 bash synthesis/fresh-confirm/run.sh evalpy` writes the eval.py CSVs, probe copies and curves into `scores/<run>/`.
4. **Write up fresh-confirm:**
   - Add a results section to its README, laying out F1–F6 against the predictions table (d3472f2). Do not rewrite the predictions.
   - Write a `make_plots.py` (it has none yet) and render the plots. Add a Figures table, then commit.
5. **Merge `synthesis/fresh-confirm` into `synthesis/base`.** Then:
   - Run `bash synthesis/final/run.sh`. `collect.py` picks up the fresh rows and `make_plots.py` draws fig3.
   - Fill the "Fresh-seed confirmation" and "Answer" sections of `synthesis/final/README.md`, and drop the DRAFT banner.
   - Update the charter's studies table.
6. **Independent review** (Opus subagent) of the `final/` claims against `scores/`. Apply fixes, then commit.
7. **Ask the user:**
   - (a) Merge the `synthesis/*` branches into `main`? Delete this file first.
   - (b) Optional: train fresh BEATs seeds (~19 h each) to confirm X1 on fresh seeds too?
8. Update memory (`project-synthesis-phase.md`, `project-ablation-board.md`, `project-ensemble-backend-devprobe.md`) with the final systems.

## Recovery

- **A run hit `FAIL train ... after 3 attempts`** (`ablation/train_one.sh` retries a crash 3×).
  - **Do not rerun `train_one.sh` for it.** Without a `done` marker it deletes `state_last.pth` and starts from scratch.
  - Resume by hand from `../OS-SCL-syn-fresh`:
    ```bash
    PY=/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python
    nohup setsid $PY train.py --gpu_num G --epoch 300 --desc fc_spec_sXXXX --state_every 10 \
      --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed XXXX --amp fp16 --cudnn_bench 1 --lean 1 \
      --cache_dir data/cache --probe_every 5 --wave_aug 1 --wave_aug_ops noise,shift --spec_branch 1 \
      --crash_resume check_points/SCLTFSTgramMFN_fc_spec_sXXXX/state_last.pth \
      >> synthesis/fresh-confirm/results/train_fc_spec_sXXXX.log 2>&1 < /dev/null &
    ```
    For an `fc_nogain` run, drop `--spec_branch 1`.
  - Afterwards write `check_points/<run>/wall.csv` (`start,end,seconds,epochs,gpu`) and `touch check_points/<run>/done`, then relaunch `post` with `nohup setsid bash synthesis/fresh-confirm/run.sh post > synthesis/fresh-confirm/results/post.log 2>&1 < /dev/null &`.
- **smartlab1 RAM is faulty** (non-ECC, byte 62 flips). Treat segfaults as hardware first and rerun.
  - Bit flips have also hit page-cache copies of `.npz` files. `score.py` evicts pages before every load and retries.
  - GPU 1 is slower (PCIe x1).
- **Never `pkill -f` a pattern that appears in your own command line.** Use anchored `pgrep -f "^..."`. Stale Claude shell wrappers carry long command lines that mention `run.sh train` and similar.

## Key numbers (Amean = mean over machines of (AUC + pAUC)/2; paper: dev 95.71/90.23/91.23, eval 93.22)

| system | dev Amean | eval Amean |
|---|---|---|
| R0 single (`eval.py`), mean of 8 | 93.59 | 93.14 |
| **C1ᵀ single, mean of 8** | 94.26 | 94.10 |
| **E2′[A8]**, reported best from scratch | 94.77 | 94.78 |
| E2′[GS12], dev pick (tie) | 95.21 | 94.93 |
| F1k (BEATs) | 95.01 | 95.25 |
| **X1 (BEATs)** | 95.37 | 95.42 |

Fresh-confirm predictions (frozen, d3472f2):

| prediction | pass rule |
|---|---|
| F1 | Fresh C1ᵀ 4-seed mean: dev ≥ 93.88 and eval ≥ 93.34 |
| F2 | C1ᵀ beats each paper number (dev AUC 95.71, pAUC 90.23, mAUC 91.23; eval 93.22) on ≥ 3/4 seeds |
| F3 | C1ᵀ − C1 on dev > 0 (reported only) |
| F4 | GS8 − G4 > 0 on dev and on eval; "confirmed" if the eval CI excludes 0 |
| F5 | Fresh GS8 level, reported against paper 93.22 and A8 94.78 |
| F6 | X1 − GS8 > 0 on eval (external) |

Exact wording: `../OS-SCL-syn-fresh/synthesis/fresh-confirm/README.md`.

Open caveat already written into `final/`: single-net C1ᵀ loses 1.9 Amean on **eval** valve vs R0
(dev is up). The ensembles don't.
