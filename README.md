# Sella Optimizer Autoresearch

This repository is an autoresearch harness for improving `sella_tiny.py`, a compact Sella-like molecular geometry optimizer. It follows the spirit of [karpathy/autoresearch](https://github.com/karpathy/autoresearch): keep the benchmark fixed, let an LLM edit one file, run the evaluation, and keep only valid improvements.

During experiments, `sella_tiny.py` is the only editable source file. The score is `fitness = mean_rel_steps`, where lower is better. A run is invalid unless `mean_rel_energy >= 0.999`; invalid runs receive `fitness = 1000.0` and must be discarded.

## Project Layout

- `sella_tiny.py` - the optimizer under research and the only file agents should edit during experiments.
- `program.md` - the main instructions for the research agent, including the metric contract and search policy.
- `ralph_autoresearch_prompt.md` - the compact prompt passed to Open Ralph Wiggum.
- `ralph_autoresearch_template.md` - the default Ralph prompt template. It intentionally omits Ralph's completion promise so the agent does not see the sentinel it would need to print to stop the loop.
- `.cursorignore` - hides `.ralph/` and `logs_autoresearch/` from Cursor Agent so Ralph internals and logs do not leak back into the research prompt.
- `validate.py` and `distributed_validate/` - the fixed Redis-backed evaluator.
- `molecules/` - train/test xTB molecule definitions and XYZ geometries.
- `scripts/` - installation, worker, Redis, and experiment helpers.
- `metrics.yaml` - machine-readable metric contract.

## Installation

From a fresh clone, install Bun, Open Ralph Wiggum, Cursor Agent, Miniforge if needed, and the `sella-autoresearch` conda environment:

```bash
scripts/install_autoresearch_tools.sh
```

If you already manage conda yourself, keep the installation narrower:

```bash
INSTALL_MINIFORGE=0 scripts/install_autoresearch_tools.sh
```

After installation, make sure the local shell can see the tools and activate the environment:

```bash
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sella-autoresearch
```

If you use an existing environment instead, it must include `ase`, `jax`, `numpy`, `scipy`, `tblite`, `redis`, `cloudpickle`, and a `redis-server` executable.

## Evaluation

Start Redis:

```bash
scripts/start_redis.sh 6379
```

Start workers:

```bash
scripts/start_workers.sh localhost 6379 8 xtb 1
```

Evaluate the current optimizer:

```bash
scripts/run_experiment.sh baseline
```

The last JSON line in `run.log` contains `fitness`, `mean_rel_steps`, `mean_rel_energy`, and `is_valid`. `scripts/run_experiment.sh` also appends a row to `results.tsv`.

On a remote worker host with this repo cloned and the conda environment active, run:

```bash
scripts/start_workers.sh <redis-host> 6379 <num-workers> xtb <xtb-threads-per-worker>
```

## Launch Autoresearch

Run the autoresearch loop from a research branch in the same checkout. The default Ralph setup assumes existing Redis and workers are already available on `localhost:6379`; it does not start or stop them.

```bash
git switch -c autoresearch/creative
rm -f results.tsv
mkdir -p logs_autoresearch
LOG="logs_autoresearch/ralph-$(date +%Y%m%d-%H%M%S).log"

tmux new-session -d -s sella-ralph-autoresearch -n ralph "bash -lc '
  set -Eeuo pipefail
  export PATH=\"\$HOME/.local/bin:\$HOME/.bun/bin:\$PATH\"
  source \"\$HOME/miniforge3/etc/profile.d/conda.sh\"
  conda activate sella-autoresearch
  ralph \
    --prompt-file ralph_autoresearch_prompt.md \
    --prompt-template ralph_autoresearch_template.md \
    --agent cursor-agent \
    --model auto \
    --max-iterations 0 \
    --completion-promise RALPH_AUTORESEARCH_NEVER_COMPLETE_$(date +%Y%m%d_%H%M%S) \
    --no-commit \
    --no-questions \
    --last-activity-timeout 3h \
    2>&1 | tee -a \"$LOG\"
'"
```

Attach to the loop with:

```bash
tmux attach -t sella-ralph-autoresearch
```

Stop it with:

```bash
tmux kill-session -t sella-ralph-autoresearch
```

Ralph relaunches Cursor Agent after each exit. The template hides the completion promise from the agent, while `.cursorignore` and the prompt tell the agent not to inspect Ralph state or log files. The agent is responsible for editing only `sella_tiny.py`, running `REDIS_PORT=6379 scripts/run_experiment.sh "short description"`, committing valid improvements, and discarding invalid or worse experiments.
