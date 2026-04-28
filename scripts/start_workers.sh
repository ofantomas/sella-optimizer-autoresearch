#!/usr/bin/env bash
set -Eeuo pipefail

REDIS_HOST="${1:-localhost}"
REDIS_PORT="${2:-6379}"
NUM_WORKERS="${3:-1}"
MODE="${4:-xtb}"
XTB_THREADS="${5:-1}"
POLL_INTERVAL="${6:-1.0}"
MAX_TASKS="${7:-300}"
LOG_DIR="${8:-logs_validate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR"
cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup EXIT

for idx in $(seq 1 "$NUM_WORKERS"); do
    "$PYTHON_BIN" -m distributed_validate.worker \
        --redis-host "$REDIS_HOST" \
        --redis-port "$REDIS_PORT" \
        --mode "$MODE" \
        --num-threads "$XTB_THREADS" \
        --poll-interval "$POLL_INTERVAL" \
        --log-dir "$LOG_DIR" \
        --worker-name "validate-worker-$idx" \
        --max-tasks "$MAX_TASKS" &
    sleep 0.2
done

wait
