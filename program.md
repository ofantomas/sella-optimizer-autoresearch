# Sella Optimizer Autoresearch

You are running an autoresearch loop for a molecular geometry optimizer. This repo follows the spirit of `karpathy/autoresearch`: the human configures this Markdown program and the agent repeatedly edits one Python file, runs the benchmark, keeps improvements, and discards regressions.

## Scope

You may edit exactly one source file:

- `sella_tiny.py`

Do not edit the evaluator, dataset, distributed worker code, scripts, metrics, environment, or this `program.md` during an experiment run. Treat every other file as the fixed benchmark harness.

## Goal

Improve the optimizer by changing `sella_tiny.py` so it needs fewer force calls while preserving essentially all baseline energy lowering.

The primary score is:

```text
fitness = mean_rel_steps
```

Lower is better. `mean_rel_steps < 1.0` means fewer force calls than the baseline on average.

A program is INVALID if:

```text
mean_rel_energy < 0.999
```

This is the most important constraint. Do not keep a faster optimizer if it loses energy quality. Invalid runs receive `fitness = 1000.0` and must be discarded, no matter how low their raw step count looks.

## Simplicity Criterion

All else being equal, simpler is better. A small valid step-count improvement that adds ugly, brittle, or hard-to-reason-about optimizer complexity is usually not worth keeping. Conversely, removing code and getting equal or better valid fitness is a strong simplification win.

When evaluating whether to keep a change, weigh complexity cost against improvement size:

- A tiny `mean_rel_steps` improvement that adds a large hacky special case is probably not worth it.
- A tiny `mean_rel_steps` improvement from deleting code, reducing branching, or making an invariant clearer is usually worth keeping.
- An approximately neutral score change that makes `sella_tiny.py` substantially simpler, more robust, or easier to mutate can be kept.

Do not confuse simplicity with conservatism. Larger architectural changes are allowed when they are coherent, measurable, and preserve the validity requirement. Avoid changes that merely hide failures, swallow errors, or add opaque constants without a clear optimizer rationale.

## Interface Contract

`sella_tiny.py` must keep this interface:

```python
def minimize_func(
    positions,
    atomic_numbers,
    calc,
    max_force_calls,
    converged,
):
    ...
    return final_positions, n_force_calls


def entrypoint():
    return minimize_func
```

The optimizer must never call `calc` more than `max_force_calls` times. It must not stop early unless `converged()` is true. Outputs must be finite and have the same shape as the input positions.

## Setup

From a fresh clone:

```bash
conda env create -f environment.yml
conda activate sella-autoresearch
```

Start a Redis server in one terminal:

```bash
scripts/start_redis.sh 6379
```

Start workers in one or more terminals or hosts:

```bash
scripts/start_workers.sh localhost 6379 8 xtb 1
```

Run a single evaluation:

```bash
scripts/run_experiment.sh baseline
```

The JSON line in `run.log` is the source of truth.

## Experiment Loop

The first experiment must be the unmodified baseline. Record it in `results.tsv`.

Loop forever:

1. Inspect current `results.tsv`, `run.log`, and recent git history.
2. Make one focused change to `sella_tiny.py`.
3. Commit the change with a short description.
4. Run `scripts/run_experiment.sh "description of idea"`.
5. Read the JSON output. Keep only if `is_valid == 1` and `fitness` is lower than the best valid score so far.
6. If invalid or worse, record it and reset back to the previous best commit.
7. Continue with the next idea.

Do not optimize for `mean_rel_energy` beyond the validity floor unless step count is tied. The target is the lowest valid `mean_rel_steps`.

This loop is autonomous. Once the experiment loop has begun after setup, do not pause to ask the human whether to continue, whether a stopping point is good, or whether the next idea is acceptable. The human may be asleep or away and expects the loop to continue indefinitely until manually interrupted.

If a change works, keep it and advance the branch. If it does not work, discard it and return to the best known valid state. Rewind sparingly, and only when the branch has clearly moved into an unproductive or broken region.

If you feel stuck, think harder instead of stopping: re-read `sella_tiny.py`, inspect prior `results.tsv` entries, look for patterns in failures, combine previous near-misses, try simplifications, or test more radical optimizer changes. The loop runs until interrupted.

## Results TSV

`scripts/run_experiment.sh` appends to `results.tsv` with columns:

```text
commit	fitness	mean_rel_steps	mean_rel_energy	converged	is_valid	status	description
```

Use `status=keep` for a new best valid result, `discard` for valid but worse, and `invalid` for anything with `is_valid != 1`.

## Search Hints

High-value areas in `sella_tiny.py` include trust-radius adaptation, Hessian update stability, initialization, line/search step restrictions, and internal-coordinate handling. Prefer simple changes that improve robustness and reduce force calls together. Avoid broad exception swallowing or fallbacks that hide real failures.
