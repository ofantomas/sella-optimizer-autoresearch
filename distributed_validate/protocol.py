from __future__ import annotations

import pickle
import time
import uuid
from typing import Iterable

QUEUE_PREFIX = "validate"
TASK_TTL_SECONDS = 3600
RESULT_TTL_SECONDS = 86400


def queue_key(mode: str) -> str:
    return f"{QUEUE_PREFIX}:queue:{mode}"


def task_key(task_id: str) -> str:
    return f"{QUEUE_PREFIX}:taskdata:{task_id}"


def result_key(task_id: str) -> str:
    return f"{QUEUE_PREFIX}:result:{task_id}"


def create_task_id(mode: str, mol_name: str) -> str:
    return (
        f"{mode}:{mol_name.replace(':', '_')}:"
        f"{uuid.uuid4().hex[:8]}"
    )


def submit_optimization_task(redis_conn, task: dict) -> str:
    task_id = create_task_id(
        mode=task["mode"],
        mol_name=task["mol_name"],
    )
    payload = dict(task)
    payload["task_id"] = task_id
    redis_conn.set(task_key(task_id), pickle.dumps(payload), ex=TASK_TTL_SECONDS)
    redis_conn.lpush(queue_key(task["mode"]), task_id)
    return task_id


def load_next_optimization_task(
    redis_conn, modes: Iterable[str]
) -> tuple[str | None, dict | None]:
    for mode in modes:
        task_id = redis_conn.rpop(queue_key(mode))
        if not task_id:
            continue
        task_id_str = task_id.decode()
        task_data = redis_conn.get(task_key(task_id_str))
        if task_data is None:
            continue
        return task_id_str, pickle.loads(task_data)
    return None, None


def store_optimization_result(redis_conn, task_id: str, result: dict) -> None:
    status = "failed" if result.get("error") else "complete"
    redis_conn.hset(
        result_key(task_id),
        mapping={
            "status": status,
            "data": pickle.dumps(result),
            "timestamp": str(time.time()),
        },
    )
    redis_conn.expire(result_key(task_id), RESULT_TTL_SECONDS)


def get_optimization_result(redis_conn, task_id: str) -> dict | None:
    key = result_key(task_id)
    if not redis_conn.exists(key):
        return None
    status = redis_conn.hget(key, "status")
    data = redis_conn.hget(key, "data")
    if status is None or data is None:
        return None
    return {
        "status": status.decode(),
        "data": pickle.loads(data),
    }


def delete_task_artifacts(redis_conn, task_id: str) -> None:
    redis_conn.delete(result_key(task_id))
    redis_conn.delete(task_key(task_id))


# Backward-compatible aliases for older imports in this repo.
submit_task = submit_optimization_task
load_next_task = load_next_optimization_task
store_result = store_optimization_result
get_result = get_optimization_result
