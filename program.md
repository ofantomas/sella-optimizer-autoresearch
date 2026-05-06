# Program: Molecule Geometry Optimization Algorithm

## Challenge

Implement a geometry optimization algorithm for drug-like molecules (up to ~100 atoms) in vacuum. Minimize total energy by updating atomic positions based on forces. Use fewer force evaluations than the baseline while recovering the same energy lowering.

## Objective

Implement a function `minimize_func` that iteratively moves atoms toward lower energy until convergence or the force-call budget is exhausted.

**Output:** `(final_positions, n_force_calls)`

## Implementation Requirements

### Libraries

Use `numpy` and `scipy`. Fix random seeds if using randomness (e.g., `np.random.seed(42)`).

### Function Interface

You must implement the following structure exactly:

```python
import numpy as np
from typing import Callable

# helpers here

def minimize_func(
    positions: np.ndarray,       # shape (N, 3), float64
    atomic_numbers: np.ndarray,  # shape (N,), integer
    calc: Callable,              # calc(pos) -> (energy, forces); forces = -grad(E)
    max_force_calls: int,        # hard budget
    converged: Callable,         # converged() -> bool; free to call (does not count toward budget)
) -> tuple[np.ndarray, int]:
    """
    Optimize molecular geometry by minimizing energy.
    # Brief description of the algorithm

    Args:
        positions: Initial atomic positions in nanometers (N, 3)
        atomic_numbers: Element atomic numbers (N,)
        calc: Energy/force calculator, returns (energy_kJ_per_mol, forces_kJ_per_mol_per_nm).
              Forces are -grad(E), i.e. point downhill.
        max_force_calls: Maximum allowed calls to calc(); set to max(50, 3*N) where N = number of atoms
        converged: Convergence check callable (free — does not count as a force call)

    Returns:
        final_positions: Optimized positions, shape (N, 3)
        n_force_calls: Actual number of calc() calls used
    """
    # Your optimization algorithm here
    return optimized_positions, n_calls_used
```

### Critical Implementation Notes

- The validation harness loads `minimize_func` directly from the module — expose it at module scope under that exact name.
- The validation function will execute `minimize_func` on multiple molecular conformations.
- Do not modify the function signature.

## Constraints

- The optimization MUST NOT terminate before either `converged()` returns `True` or `max_force_calls` is reached. Stopping early is the most common failure mode.
- Hard stop at `max_force_calls`: exceeding it makes the run invalid.
- If energy or forces are non-finite, treat the step as failed (reject or abort). Accepting non-finite energy causes downstream errors.
- Output positions must be finite; `NaN` or `Inf` cause the run to be marked invalid.

## Scoring

For each test conformation the evaluator computes:

- `rel_steps_i = n_force_calls / n_steps_baseline`
- `rel_energy_i = (E_start - E_final) / (E_start - E_final_baseline)`
where `E_final_baseline` is the energy reached by the baseline algorithm on the same conformation.

The aggregate metrics across all conformations (`mean_rel_steps`, `mean_rel_energy`, `converged`, `is_valid`) are defined in the table below.

## Metrics

| Name                       | Description                                                                                                | Direction        | Decimals | Significant Change | Sentinel |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------- | -------- | ------------------ | -------- |
| `mean_rel_steps` (primary) | Mean ratio `n_force_calls / baseline_n_force_calls` across molecules.                                      | lower is better  | 6        | 0.01               | 1000     |
| `mean_rel_energy`          | Mean ratio `(E_start - E_final) / (E_start - E_final_baseline)`. Must be >= 0.999 for the run to be valid. | higher is better | 6        | 0.001              | -1       |
| `converged`                | Fraction of test conformations where `converged()` returned `True` before `max_force_calls` was exhausted. | higher is better | 4        | 0.01               | -1       |
| `is_valid`                 | 1 if `mean_rel_energy >= 0.999` and no `NaN`/`Inf` in positions, 0 otherwise.                              | higher is better | 0        | 1.0                | -1       |

## Convergence Criteria

The evaluator tracks convergence via the `converged()` callable passed to `minimize_func`. Calling `converged()` returns `True` when the last `calc()` call met all of:

1. **Energy change:** `|E - E_prev| < 0.013 kJ/mol` (5e-6 Hartree)
2. **Max gradient:** `max|F_i| < 14.9 kJ/mol/nm` (3e-4 Eh/Bohr)
3. **RMS gradient:** `rms(F_i) < 4.96 kJ/mol/nm` (1e-4 Eh/Bohr)
4. **Max displacement:** `max|r_i - r_prev_i| < 0.000212 nm` (4e-3 Bohr)
5. **RMS displacement:** `rms(r_i - r_prev_i) < 0.000106 nm` (2e-3 Bohr)

All five must be satisfied simultaneously. The first `calc()` call always returns `converged() == False` (no previous step to compare against).

## Autoresearch Interface

The autoresearch loop iterates on `algo.py` — the only source file it is allowed to modify. The loop also maintains three append-only artifacts — `run.log` (written by `validate.py`), `results.tsv`, and `full_log.md` (both written by the loop). All three live in the working tree and are **never committed**; only `algo.py` is committed during the loop. No other files may be touched by the loop.

### Setup

- **Create the working branch** `exp/autoresearch` from the `autoresearch` branch. It must not already exist — this is a fresh run; if it does, delete or rename the old one before starting.
  ```bash
  git checkout autoresearch
  git checkout -b exp/autoresearch
  ```
- Confirm the remote Redis is reachable through the SSH tunnel to `airi220` (see `validate.py` header) before starting the loop:
  ```bash
  redis-cli -h 127.0.0.1 -p 6382 ping
  ```
  Expect `PONG`. If `redis-cli` isn't installed locally, use:
  ```bash
  python -c "import redis; print(redis.Redis(host='127.0.0.1', port=6382).ping())"
  ```
  which prints `True` on success. Anything else (connection refused, timeout) means the tunnel is down — open it with `ssh -L 6380:127.0.0.1:6382 -N airi220` in a separate terminal before continuing.

### Running the loop

LOOP FOREVER:

1. **Inspect git state** — confirm the branch is `exp/autoresearch` and note the current `HEAD`.
2. **Edit `algo.py`** with one experimental idea — a single coherent change per cycle, direct code edit.
3. **Commit** before running, so the SHA is stable when recorded:
  ```bash
   git add algo.py
   git commit -m "cycle <N>: <short description>"
  ```
4. **Run** `python validate.py`. All output goes into `run.log`; do not pipe through `tee` or read stdout.
5. **Parse** `run.log`:
  - Look for `is_valid = 1` and the four combined metrics.
  - If the block ends with `outcome: invalid (no results)` or contains `error:` lines, the run crashed or erred.
6. **Record one row in `results.tsv`** — tab-separated, schema below. Crashes and rejects are recorded too.
7. **Append a narrative section to `full_log.md`** using the template in its section below.
8. **Decide — keep, discard, or crash:**
  - **keep** (advance the branch): run is valid AND `mean_rel_steps` improved by at least `significant_change` (0.01) vs. the most recent kept cycle. Leave the commit in place.
  - **discard** (valid run but no improvement): `git reset --hard HEAD~1` to undo the commit. The branch tip stays where it was before this cycle.
  - **crash** (invalid or errored): `git reset --hard HEAD~1` as above. Note the distinction in `full_log.md` so future cycles don't repeat the broken idea.
  The baseline for "improved" is the `mean_rel_steps` of the most recent *kept* cycle in `results.tsv`. The very first cycle establishes the baseline and is kept if valid, regardless of the improvement threshold.

Go back to step 1.

### Research strategy: analysis over tuning

**Do NOT spend cycles on hyperparameter tuning.** Twiddling constants (step sizes, trust-radius bounds, tolerance thresholds, line-search coefficients, damping factors, etc.) is explicitly out of scope — it produces noisy, non-generalizable wins and burns the budget. If a cycle's only change is "raise X from 0.3 to 0.5" with no structural justification, don't run it.

**Instead, spend the bulk of each cycle on deep per-molecule analysis.** Before proposing a change, reason from the data:

- **Per-molecule metrics in `run.log`.** Read the full per-molecule table, not just the combined aggregates. Which molecules are cheap wins, which are outliers, which regressed, which fail to converge? Cluster the outcomes — e.g. "all molecules that regressed have >60 atoms", "failures all contain sulfur", "the worst 10 % of `rel_steps` share a flexible ring".
- **Geometry from `molecules/xyz/{key}_mm.xyz`.** For any molecule that regressed, failed to converge, or looks anomalous, open its `.xyz` file and inspect the structure: atom count, element composition, obvious features (rings, long chains, floppy side groups, close contacts, heavy atoms). Structural features are the actual causal variable behind an optimizer's behavior — use them to hypothesize.
- **Baseline context in `molecules/train_XTB.json`.** Each key maps to baseline metadata: initial/final Hartree energies, `n_steps`, convergence flag. Use these to sanity-check what "good" looks like on a given molecule (`E_start - E_final_baseline` tells you the depth of the basin; `n_steps` tells you how hard ORCA found it). A molecule where the baseline itself took 200 steps is not the same problem as one it solved in 15.

Every Hypothesis in `full_log.md` should cite at least one concrete observation from these three sources (e.g. *"in cycles 7 and 12 the five largest molecules — `mol_N_`* with N >= 80 atoms — all regressed; their `.xyz` files show long aliphatic chains, so the trust-radius cap likely under-steps along soft directions"*). Changes motivated this way target a specific failure mode; changes motivated by "let's try 0.4 instead of 0.3" do not.

### Timeout

Typical wall time with ~48 workers on the xTB molecule set is ~5 minutes. **If `validate.py` has not returned after 15 minutes, kill it and treat the cycle as a crash** — the remote workers, the tunnel, or Redis are in a bad state and that investigation comes before any further experiments.

Kill the hung run with:

```bash
pkill -f 'python validate.py'
```

### Crashes

Every run is recorded in `results.tsv` and `full_log.md` — even crashes and infra failures. Use judgement for what to do *after* recording:

- **Local, obvious fix** (typo, missing import, off-by-one in `algo.py`): amend the fix in-place, commit, re-run. One or two fix-and-retry attempts max — beyond that, it's not a quick fix.
- **Infrastructure failure** (tunnel dropped, Redis unreachable, all workers crashed): record the crash row, then pause the loop and fix the infra before the next cycle. The `full_log.md` section should note "infra failure" in the Insight so the row isn't mistaken for an experiment result.
- **The idea itself is broken** (NaN gradients on every molecule, logic error inherent to the change): record the crash row, reset, move on.

### Never stop

Once the loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep or away and expects autonomous operation until manually interrupted.

If you run out of ideas, think harder: re-read per-molecule metrics, their structure into `molecules/xyz/`, look for near-misses in earlier cycles in `full_log.md`, combine two previously-reverted ideas that failed for different reasons, or attempt a more radical change. The loop runs until the human interrupts you.

### `run.log`

Append-only log. `validate.py` writes nothing to stdout; all its output goes here. After every run it appends one block regardless of the loop's accept/revert decision. Blocks are separated by a `=== {timestamp} ===` header and contain, in order:

- Header line: `=== {iso_timestamp} ===`
- `optimizer:` — description of the loaded optimizer spec
- `mode:` / `molecules:` / `duration:` / `results:` / `errors:` — run summary
- Per-error lines (one per failed molecule), if any
- **Per-molecule** table with columns `molecule`, `n_steps`, `max_steps`, `rel_steps`, `rel_energy`, `conv`
- **Combined** metrics: `mean_rel_steps`, `mean_rel_energy`, `converged`, `is_valid`

Invalid runs (no results at all) are logged with `outcome: invalid (no results)` in place of the two tables.

### `results.tsv`

One row per completed cycle. **Tab-separated, not comma-separated** — commas are legal inside `commit_description`, so comma-separation would break. Plain `\t` between values, no pipe characters, no padding. Columns in order:

```text
cycle	timestamp	commit	accepted	mean_rel_steps	mean_rel_energy	converged	is_valid	duration_s	commit_description
```

- `commit` — short git SHA (7 chars) of the `algo.py` version that produced these metrics.
- `accepted` — `1` if kept, `0` if reset. Combined with `is_valid`, this encodes the three cycle statuses:
  - keep  → `accepted=1`, `is_valid=1`
  - discard → `accepted=0`, `is_valid=1`
  - crash → `accepted=0`, `is_valid=0`
- `commit_description` — the one-line commit message (last column; may contain spaces and commas, never tabs).

Rejected and crashed cycles are still recorded — the file is the full history of attempts, not only wins. `results.tsv` is a loop artifact and is **not** committed to git.

### `full_log.md`

Human-readable narrative log, written by the loop (not by `validate.py`). One Markdown section appended per cycle, in chronological order. The human reader consumes this to understand what was tried and why — so write in prose, not just numbers.

Each cycle's section uses this template:

```markdown
## Cycle <N> — <short title of the change> (<ISO timestamp>)

**Status:** keep | discard | crash
**Commit:** `<short sha>` (if kept)

### Hypothesis
One or two sentences: what did we expect this change to do, and why?

### Change
What was modified in `algo.py` (function(s) touched, nature of the edit — e.g. "raised `_TRUST_INIT` from 0.3 to 0.5", "replaced numerical dihedral B-row with analytic derivatives"). Reference function names so future cycles can grep.

### Results
- `mean_rel_steps`:  <prev> → <new>  (Δ = ...)
- `mean_rel_energy`: <prev> → <new>
- `converged`:       <prev> → <new>
- `is_valid`:        <new>
- Molecules that regressed or errored: <list with brief note>. A molecule counts as **regressed** when its own `rel_steps` increased by >= `significant_change` (0.01) vs. the most recent kept cycle's value for that same molecule. Errored = appeared in the `error:` lines of `run.log`.

### Insight
What did we actually learn? Did the hypothesis hold? Any surprise in the per-molecule breakdown (e.g. "small molecules improved, large ones regressed — suggests the change helps when X but not when Y")? If discarded or crashed, say why the change didn't work so future cycles don't repeat it.

### Next ideas
One or two candidate directions seeded by this cycle's outcome.
```

Keep each section concise but substantive — aim for enough context that, reading the file cold months later, someone could reconstruct the line of investigation without opening git history or `run.log`. If a cycle is a trivial tweak, the Hypothesis/Change/Insight can be one sentence each; don't pad.
