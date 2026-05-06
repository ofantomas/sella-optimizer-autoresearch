# Ralph Autoresearch Loop - Iteration {{iteration}} / {{max_iterations}}

You are running in a continuous autoresearch loop. There is no completion condition in this prompt. Keep making progress until the human manually stops the process.

Do not inspect Ralph internals. Never read or search `.ralph/`, Ralph state files, Ralph history files, or Ralph context files.

## Task

{{prompt}}

## Context From Previous Iterations

{{context}}

## Operating Instructions

1. Read the repository files needed for the optimizer task, especially `program.md`, `results.tsv`, `full_log.md`, `run.log`, git history, and `algo.py`.
2. Infer the current best valid optimizer state from git history, `results.tsv`, and `full_log.md`.
3. Make one focused research change at a time, editing only `algo.py`.
4. Run the requested validation command and record the cycle in `results.tsv` and `full_log.md`.
5. If a change is invalid, neutral, or worse, reset back to the current best state and try a different idea.
6. After each kept, discarded, or crashed experiment, immediately continue with the next experiment.

Important: do not claim the task is complete. Improvement is not completion. A clean working tree is not completion. A new best score is not completion. Continue researching.
