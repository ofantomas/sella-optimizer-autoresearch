# Sella Optimizer Autoresearch

Autoresearch-style benchmark repo for improving `sella_tiny.py`, a compact Sella-like molecular geometry optimizer. The workflow is based on the idea in [karpathy/autoresearch](https://github.com/karpathy/autoresearch): keep the benchmark fixed, let an LLM edit one file, run the evaluation, and keep only improvements.

## Objective

Only `sella_tiny.py` is editable during experiments.

The score is `fitness = mean_rel_steps`, where lower is better. A run is invalid unless `mean_rel_energy >= 0.999`. Invalid runs receive `fitness = 1000.0`.

## Files

- `sella_tiny.py` - the only optimizer file agents should edit.
- `validate.py` - fixed evaluator that submits xTB molecular optimization tasks to Redis-backed workers.
- `distributed_validate/` - client, worker, protocol, and optimizer serialization utilities.
- `molecules/` - train/test xTB molecule definitions and XYZ geometries.
- `scripts/` - Redis, worker, and experiment helpers.
- `program.md` - instructions for the LLM autoresearch loop.
- `metrics.yaml` - machine-readable metric contract.

## Environment

```bash
conda env create -f environment.yml
conda activate sella-autoresearch
```

If you already have the original validation environment, use it instead as long as it includes `ase`, `jax`, `numpy`, `scipy`, `tblite`, `redis`, and `cloudpickle`.

## Run Locally

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

The last JSON line in `run.log` contains `fitness`, `mean_rel_steps`, `mean_rel_energy`, and `is_valid`.

## Remote Workers

On another host with this repo cloned and the conda environment active, run:

```bash
scripts/start_workers.sh <redis-host> 6379 <num-workers> xtb <xtb-threads-per-worker>
```

For SSH-tunneled workers, adapt `scripts/babysit_validate.sh` from the original benchmark or start an SSH tunnel manually and point workers to the forwarded Redis port.
