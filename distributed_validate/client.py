from __future__ import annotations

import time

from distributed_validate.protocol import (
    delete_task_artifacts,
    get_optimization_result,
    submit_optimization_task,
)


class RemoteOptimizationClient:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        poll_interval_seconds: float = 0.1,
    ):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "redis package is required for distributed validation"
            ) from exc

        self._redis = redis.Redis(host=redis_host, port=redis_port)
        try:
            self._redis.ping()
        except redis.exceptions.RedisError as exc:
            raise RuntimeError(
                f"Redis is unavailable at {redis_host}:{redis_port}"
            ) from exc
        self.poll_interval_seconds = poll_interval_seconds

    def submit(self, task: dict) -> str:
        return submit_optimization_task(self._redis, task)

    def wait_for_result(self, task_id: str, timeout_seconds: float) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while True:
            payload = get_optimization_result(self._redis, task_id)
            if payload is not None:
                delete_task_artifacts(self._redis, task_id)
                result = payload["data"]
                if payload["status"] != "complete":
                    raise RuntimeError(
                        f"Worker failed for task {task_id}: "
                        f"{result.get('error', 'unknown error')}"
                    )
                return result

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for distributed result for {task_id}")

            time.sleep(self.poll_interval_seconds)


# Backward-compatible alias for any existing local imports.
RemoteComputeClient = RemoteOptimizationClient
