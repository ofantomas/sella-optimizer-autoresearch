Read and follow `program.md`.

You are running inside Open Ralph Wiggum. Ralph will relaunch this prompt repeatedly, so resume from git history, `run.log`, `results.tsv`, and the current working tree.

Runtime contract:
- Work on the current branch.
- Use the conda environment selected by the launcher.
- Use existing Redis/workers on `localhost:6379`; do not start, stop, or restart Redis/workers.
- Run evaluations with `REDIS_PORT=6379 scripts/run_experiment.sh "short description"`.

Important operating rules:
- Before the first experiment on a clean branch, evaluate the unmodified baseline and wait for it to finish.
- During experiments, edit only `sella_tiny.py`.
- Prefer creative optimizer mechanisms over scalar hyperparameter sweeps, while keeping the simplicity criterion from `program.md`.
- Keep only valid improvements to `fitness`; discard invalid runs and regressions.
- Continue indefinitely until the human manually stops the loop.

Never output the Ralph completion promise unless explicitly instructed by the human.
