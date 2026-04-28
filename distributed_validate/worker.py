from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

from convergence import init_convergence_state, is_converged, update_convergence_state
from distributed_validate.optimizer import (
    describe_optimizer_spec,
    load_minimize_func,
    optimizer_cache_key,
)
from distributed_validate.protocol import (
    load_next_optimization_task,
    store_optimization_result,
)

def str_to_bool(value):
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def setup_logging(component_name, verbose=False, log_dir="logs", log_file=None):
    """Configure logging with file handler."""
    logger = logging.getLogger(component_name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{component_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger


def get_time():
    return datetime.now().strftime("%H:%M:%S")


logger = logging.getLogger("validate_worker")
_SYSTEM_CACHE: dict[tuple[str, str, str, int], object] = {}
_OPTIMIZER_CACHE: dict[tuple[str, str | None, str, bool], object] = {}


def _configure_jax_x64() -> None:
    # Serialized Sella optimizers are unpickled in fresh worker processes, so
    # the worker must enable JAX x64 before any such payload loads JAX.
    os.environ["JAX_ENABLE_X64"] = "True"


def _backend_module_name(mode: str) -> str:
    if mode == "xtb":
        return "xtb_molecular_system"
    if mode == "ff":
        return "molecular_system"
    raise ValueError(f"Invalid mode: {mode}")


def _build_system(mode: str, baseline: dict, num_threads: int):
    module_name = _backend_module_name(mode)
    module = importlib.import_module(module_name)
    if mode == "xtb":
        return module.MolecularSystem(baseline["xyz_path"], num_threads=num_threads)
    return module.MolecularSystem(baseline["system_xml_path"])


def _get_system(mode: str, baseline: dict, num_threads: int):
    cache_key = (mode, baseline["xyz_path"], int(num_threads))
    system = _SYSTEM_CACHE.get(cache_key)
    if system is None:
        system = _build_system(mode, baseline, num_threads)
        _SYSTEM_CACHE[cache_key] = system
    return system


def _preflight_backend_import(mode: str) -> None:
    module_name = _backend_module_name(mode)
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import backend module {module_name!r} for mode {mode!r} "
            f"with interpreter {sys.executable}: {exc}"
        ) from exc


def _get_minimize_func(optimizer_spec: dict) -> object:
    cache_key = optimizer_cache_key(optimizer_spec)
    minimize_func = _OPTIMIZER_CACHE.get(cache_key)
    if minimize_func is None:
        minimize_func = load_minimize_func(optimizer_spec)
        _OPTIMIZER_CACHE[cache_key] = minimize_func
    return minimize_func


def _run_optimization_task(task: dict) -> dict:
    mode = task["mode"]
    mol_name = task["mol_name"]
    baseline = task["baseline"]
    max_steps = int(task["max_steps"])
    optimizer_spec = task["optimizer_spec"]
    num_threads = int(task.get("num_threads", 1))

    system = _get_system(mode=mode, baseline=baseline, num_threads=num_threads)
    minimize_func = _get_minimize_func(optimizer_spec)

    if mode == "xtb":
        atomic_numbers = system.atomic_numbers
        conf_pos = system.initial_positions
    else:
        from utils import parse_xyz
        atomic_numbers, conf_pos = parse_xyz(baseline["xyz_path"])

    initial_energy = system.compute(conf_pos)[0]
    state = init_convergence_state(mode, conf_pos)
    n_steps = 0

    def calc(pos: np.ndarray, conv_check: bool = False):
        nonlocal n_steps
        if conv_check:
            return is_converged(state)

        if n_steps >= max_steps:
            raise RuntimeError(
                f"Optimizer exceeded budget of {max_steps} force calls"
            )
        n_steps += 1
        pos_arr = np.asarray(pos, dtype=float).copy()
        energy, forces = system.compute(pos_arr)
        update_convergence_state(state, pos_arr, energy, forces)
        return energy, forces

    def converged_from_calc(*_a, **_k):
        return is_converged(state)

    opt_pos, n_steps_func = minimize_func(
        conf_pos.copy(),
        atomic_numbers,
        calc,
        max_force_calls=max_steps,
        converged=converged_from_calc,
    )
    opt_pos_arr = np.asarray(opt_pos, dtype=float)
    final_energy = system.compute(opt_pos_arr)[0]

    if n_steps != n_steps_func:
        logger.warning(
            "Force-call mismatch for %s: counted=%s returned=%s",
            mol_name,
            n_steps,
            n_steps_func,
        )
        n_steps = n_steps_func

    if not math.isfinite(final_energy):
        raise ValueError("Final energy is not finite")

    if opt_pos_arr.ndim != 2 or opt_pos_arr.shape != conf_pos.shape:
        raise ValueError("Wrong number of atoms or coordinates")

    energy_improvement = initial_energy - final_energy
    baseline_improvement = baseline["improvement"]
    baseline_steps = baseline["n_steps"]

    rel_energy = (
        energy_improvement / baseline_improvement if energy_improvement > 0 else 0.0
    )
    rel_steps = n_steps / baseline_steps

    return {
        "result": {
            "mol_name": mol_name,
            "converged": is_converged(state),
            "max_steps": max_steps,
            "n_steps": n_steps,
            "rel_energy": rel_energy,
            "rel_steps": rel_steps,
        },
        "error": None,
    }


def start_worker(
    redis_host: str,
    redis_port: int,
    supported_modes: list[str],
    num_threads: int = 1,
    poll_interval_seconds: float = 1.0,
    verbose: bool = False,
    log_dir: str = "logs",
    worker_name: str | None = None,
    max_tasks: int = 0,
) -> None:
    global logger
    _configure_jax_x64()
    worker_name = worker_name or f"validate_worker_{os.uname().nodename}_{os.getpid()}"
    logger = setup_logging(worker_name, verbose=verbose, log_dir=log_dir)

    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "redis package is required for distributed validation"
        ) from exc

    redis_conn = redis.Redis(host=redis_host, port=redis_port)
    try:
        redis_conn.ping()
    except redis.exceptions.RedisError as exc:
        raise RuntimeError(
            f"Redis is unavailable at {redis_host}:{redis_port}"
        ) from exc

    logger.info(
        "Starting validate worker %s for modes=%s",
        worker_name,
        ",".join(supported_modes),
    )
    logger.info("Python executable: %s", sys.executable)
    logger.info("Configured xTB threads per worker: %s", num_threads)

    if "xtb" in supported_modes:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

    for mode in supported_modes:
        _preflight_backend_import(mode)
        logger.info("Backend import check passed for mode=%s", mode)

    tasks_done = 0

    while True:
        task_id, task = load_next_optimization_task(redis_conn, supported_modes)
        if task_id is None or task is None:
            time.sleep(poll_interval_seconds)
            continue

        logger.info(
            "Processing optimization task %s for %s %s using %s (xtb_threads=%s)",
            task_id,
            task["mode"],
            task["mol_name"],
            describe_optimizer_spec(task["optimizer_spec"]),
            num_threads,
        )

        try:
            task = {**task, "num_threads": num_threads}
            result = _run_optimization_task(task)
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            logger.debug(traceback.format_exc())
            result = {
                "result": None,
                "error": str(exc),
            }

        store_optimization_result(redis_conn, task_id, result)

        # Bound native-memory growth by exiting cleanly after a fixed
        # number of tasks; the launcher respawns us.
        tasks_done += 1
        if max_tasks and tasks_done >= max_tasks:
            logger.info(
                "Processed %d tasks (limit %d); exiting for respawn.",
                tasks_done,
                max_tasks,
            )
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed validation worker")
    parser.add_argument(
        "--redis-host", type=str, default="localhost", help="Redis hostname or IP"
    )
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ff", "xtb", "both"],
        default="both",
        help="Worker backend to serve",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="xTB threads to use per worker process",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Sleep duration when no tasks are available",
    )
    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        default=False,
        help="Enable verbose (debug) logging (true/false)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to store log files"
    )
    parser.add_argument("--worker-name", type=str, default=None, help="Worker name")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help=(
            "Exit cleanly after processing this many tasks "
            "(0 = unbounded). Used with a launcher-side restart loop to "
            "bound per-worker memory."
        ),
    )

    args = parser.parse_args()
    supported_modes = ["ff", "xtb"] if args.mode == "both" else [args.mode]
    start_worker(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        supported_modes=supported_modes,
        num_threads=args.num_threads,
        poll_interval_seconds=args.poll_interval,
        verbose=args.verbose,
        log_dir=args.log_dir,
        worker_name=args.worker_name,
        max_tasks=args.max_tasks,
    )
