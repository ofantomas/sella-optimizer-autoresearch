Follow the autoresearch protocol in `program.md`: one cycle per iteration — edit `algo.py`, run `validate.py`, record the cycle in `results.tsv` and `full_log.md`, then keep/discard/crash per the rules in program.md.

You may be relaunched with this prompt repeatedly, so resume from git history, `run.log`, `results.tsv`, and the current working tree.

Never read or search `.ralph/`, Ralph state files, Ralph history files, or Ralph context files.

Runtime contract:
- Work on the current branch.
- Use the conda environment selected by the launcher.
- Use the existing Redis and babysat workers on `localhost:6379`.
- Do not start, stop, restart, kill, or babysit Redis or validation workers. The launcher owns that infrastructure.
- Run evaluations with `REDIS_PORT=6379 python validate.py`. `validate.py` appends all output to `run.log` and writes nothing to stdout.

Important operating rules:
- Before the first experiment on a clean branch, evaluate the unmodified baseline and wait for it to finish.
- During experiments, edit only `algo.py`.
- Maintain `results.tsv` and `full_log.md` exactly as described in `program.md`; do not commit them.
- Prefer per-molecule analysis and novel optimizer mechanisms over hyperparameter sweeps.
- Keep only valid improvements per the `program.md` criteria; discard valid non-improvements and crashes.
- Continue indefinitely until the human manually stops the loop.
