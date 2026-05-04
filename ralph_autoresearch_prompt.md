Read and follow `program.md`.

You are running inside Open Ralph Wiggum. Ralph will relaunch this prompt repeatedly, so resume from git history, `run.log`, `results.tsv`, and the current working tree.

Do not inspect Ralph internals. Never read or search `.ralph/`, Ralph state files, Ralph history files, or Ralph context files.

Runtime contract:
- Work on the current branch.
- Use the conda environment selected by the launcher.
- Use the existing Redis and babysat workers on `localhost:6379`.
- Do not start, stop, restart, kill, or babysit Redis or validation workers. The launcher owns that infrastructure.
- Run evaluations with `REDIS_PORT=6379 scripts/run_experiment.sh "short description"`.

Important operating rules:
- Before the first experiment on a clean branch, evaluate the unmodified baseline and wait for it to finish.
- During experiments, edit only `sella_tiny.py`.
- Prefer creative optimizer mechanisms over scalar hyperparameter sweeps, while keeping the simplicity criterion from `program.md`.
- Keep only valid improvements to `fitness`; discard invalid runs and regressions.
- Continue indefinitely until the human manually stops the loop.
