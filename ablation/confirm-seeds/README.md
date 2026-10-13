# Confirm seeds — fresh paper-model checkpoints for confirmation

> **Status: training (started 2026-10-13 17:19). Nothing scored, R0 included.**
>
> **Resume:**
> - Check progress: `tail -c 300 ablation/confirm-seeds/results/train_cs_relu_max_s<seed>.log` and `results/run_train.log`.
> - When training ends, `run.sh train` copies each `config.json` and distils the curves.
> - **Before `run.sh score`, write and commit the predictions in this README.** The score stage is a stub until then.

## Why this exists

The open single-model rules have only been checked on checkpoints whose eval
scores we have already seen:

- S1′ from `../ensemble-backend`;
- F_A1 from `../memory-bank`, if adopted;
- an 8-member E1.

A clean confirmation needs new checkpoints. These are 4 more seeds of the
paper's model, trained with the exact recipe of `../tfgram-linear`'s `relu_max`
arm (`--num_workers 8`, 300 epochs, same snapshots). Only the seed and desc
differ.

| run | seed | GPU |
|-----|------|-----|
| `SCLTFSTgramMFN_cs_relu_max_s2028` | 2028 | 2 |
| `SCLTFSTgramMFN_cs_relu_max_s2029` | 2029 | 1 |
| `SCLTFSTgramMFN_cs_relu_max_s2030` | 2030 | next free |
| `SCLTFSTgramMFN_cs_relu_max_s2031` | 2031 | next free |

## Pre-registration

*To be written before any score is computed on these checkpoints.*
