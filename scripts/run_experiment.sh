#!/usr/bin/env bash
set -Eeuo pipefail

DESCRIPTION="${1:-manual}"
RUN_LOG="${RUN_LOG:-run.log}"
PROGRAM="${PROGRAM:-sella_tiny.py}"
SPLIT="${SPLIT:-train}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

python validate.py \
    --program "$PROGRAM" \
    --split "$SPLIT" \
    --redis-host "$REDIS_HOST" \
    --redis-port "$REDIS_PORT" \
    > "$RUN_LOG" 2>&1

python scripts/score_log.py "$RUN_LOG"

if [ ! -f results.tsv ]; then
    printf "commit\tfitness\tmean_rel_steps\tmean_rel_energy\tconverged\tis_valid\tstatus\tdescription\n" > results.tsv
fi

python - "$RUN_LOG" "$DESCRIPTION" <<'INNER_PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
description = sys.argv[2].replace("\t", " ").replace("\n", " ")
result = None
with open(log_path, "r", encoding="utf-8") as file_obj:
    for line in file_obj:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            result = json.loads(line)
if result is None:
    raise SystemExit(f"No JSON result found in {log_path}")

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
if int(result["is_valid"]) != 1:
    status = "invalid"
else:
    best = None
    results_path = Path("results.tsv")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as file_obj:
            next(file_obj, None)
            for line in file_obj:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 7 and fields[6] == "keep":
                    fitness = float(fields[1])
                    best = fitness if best is None else min(best, fitness)
    status = "keep" if best is None or result["fitness"] < best else "discard"
row = (
    f"{commit}\t{result['fitness']:.6f}\t{result['mean_rel_steps']:.6f}\t"
    f"{result['mean_rel_energy']:.6f}\t{result['converged']:.4f}\t"
    f"{result['is_valid']}\t{status}\t{description}\n"
)
with open("results.tsv", "a", encoding="utf-8") as file_obj:
    file_obj.write(row)
INNER_PY
