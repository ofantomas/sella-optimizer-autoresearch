from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from distributed_validate.client import RemoteOptimizationClient
from distributed_validate.optimizer import (
    describe_optimizer_spec,
    load_minimize_func,
    normalize_optimizer_spec,
)
from utils import HARTREE_TO_KJ

REPO_ROOT = Path(__file__).resolve().parent
ENERGY_VALIDITY_FLOOR = 0.999
INVALID_FITNESS = 1000.0


def _load_module_from_path(module_name: str, path: str | os.PathLike[str]):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def serialize_program_entrypoint(program_path: str | os.PathLike[str]) -> bytes:
    try:
        import cloudpickle
    except ImportError as exc:
        raise RuntimeError("cloudpickle is required to serialize sella_tiny.py for workers") from exc

    module_path = Path(program_path).resolve()
    module_name = f"_sella_candidate_{abs(hash(str(module_path))):x}"
    module = _load_module_from_path(module_name, module_path)
    entrypoint = getattr(module, "entrypoint", None)
    if entrypoint is None or not callable(entrypoint):
        raise RuntimeError(f"{program_path} must define callable entrypoint()")
    minimize_fn = entrypoint()
    if minimize_fn is None or not callable(minimize_fn):
        raise RuntimeError("entrypoint() must return a callable optimizer")

    if hasattr(cloudpickle, "register_pickle_by_value"):
        cloudpickle.register_pickle_by_value(module)
    return cloudpickle.dumps(minimize_fn)


def load_program_entrypoint(program_path: str | os.PathLike[str]) -> Callable:
    module = _load_module_from_path("sella_candidate", program_path)
    entrypoint = getattr(module, "entrypoint", None)
    if entrypoint is None or not callable(entrypoint):
        raise RuntimeError(f"{program_path} must define callable entrypoint()")
    minimize_fn = entrypoint()
    if minimize_fn is None or not callable(minimize_fn):
        raise RuntimeError("entrypoint() must return a callable optimizer")
    return minimize_fn


def discover_xtb_molecules(molecules_dir: str | os.PathLike[str], split: str) -> tuple[list[str], dict]:
    molecules_path = Path(molecules_dir)
    baselines_path = molecules_path / f"{split}_XTB.json"
    with open(baselines_path, "r", encoding="utf-8") as file_obj:
        baselines_raw = json.load(file_obj)

    mol_cost: dict[str, float] = {}
    mol_names: list[str] = []
    baselines: dict[str, dict] = {}
    for mol_name, metrics in baselines_raw.items():
        mol_names.append(mol_name)
        baselines[mol_name] = {
            "n_steps": metrics["n_steps"],
            "improvement": (metrics["initial_energy"] - metrics["final_energy"]) * HARTREE_TO_KJ,
            "n_atoms": metrics["n_atoms"],
            "xyz_path": str(molecules_path / "xyz" / f"{mol_name}_mm.xyz"),
        }
        mol_cost[mol_name] = metrics["n_steps"] * metrics["n_atoms"]

    molecules = sorted(mol_names, key=lambda name: mol_cost[name], reverse=True)
    return molecules, baselines


def _invalid_score(reason: str) -> dict:
    return {
        "fitness": INVALID_FITNESS,
        "mean_rel_steps": INVALID_FITNESS,
        "mean_rel_energy": -1.0,
        "converged": -1.0,
        "is_valid": 0,
        "lower_is_better": True,
        "invalid_reason": reason,
    }


def score_results(results: list[dict], num_errors: int) -> dict:
    if len(results) == 0 or num_errors > 0:
        return _invalid_score(f"results={len(results)} errors={num_errors}")

    if any(result["n_steps"] > result["max_steps"] for result in results):
        return _invalid_score("exceeded max force-call budget")

    if any((result["n_steps"] < result["max_steps"]) and not result["converged"] for result in results):
        return _invalid_score("stopped before convergence")

    mean_rel_steps = float(np.mean([result["rel_steps"] for result in results]))
    mean_rel_energy = float(np.mean([result["rel_energy"] for result in results]))
    converged = float(sum(result["converged"] for result in results) / len(results))
    is_valid = int(mean_rel_energy >= ENERGY_VALIDITY_FLOOR)

    return {
        "fitness": mean_rel_steps if is_valid else INVALID_FITNESS,
        "mean_rel_steps": mean_rel_steps,
        "mean_rel_energy": mean_rel_energy,
        "converged": converged,
        "is_valid": is_valid,
        "lower_is_better": True,
        "invalid_reason": "" if is_valid else f"mean_rel_energy < {ENERGY_VALIDITY_FLOOR}",
    }


class Evaluator:
    def __init__(
        self,
        molecules_dir: str | os.PathLike[str] = REPO_ROOT / "molecules",
        split: str = "train",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        poll_interval_seconds: float = 0.1,
        task_timeout_xtb: float = 3600.0,
    ):
        self.molecules_dir = str(molecules_dir)
        self.split = split
        self.molecules, self.baselines = discover_xtb_molecules(self.molecules_dir, split)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.poll_interval_seconds = poll_interval_seconds
        self.task_timeout_xtb = task_timeout_xtb

    def evaluate(self, optimizer: Callable | str | dict[str, Any], verbose: bool = False) -> dict:
        optimizer_spec = normalize_optimizer_spec(optimizer)
        load_minimize_func(optimizer_spec)
        if verbose:
            print(f"Optimizer: {describe_optimizer_spec(optimizer_spec)}")
            print(f"Split: {self.split}, molecules: {len(self.molecules)}")

        cases = []
        for mol_name in self.molecules:
            baseline = self.baselines[mol_name]
            cases.append(
                {
                    "mol_name": mol_name,
                    "baseline": baseline,
                    "max_steps": max(50, 3 * baseline["n_atoms"]),
                }
            )

        client = RemoteOptimizationClient(
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            poll_interval_seconds=self.poll_interval_seconds,
        )

        submitted: list[tuple[str, str]] = []
        for case in cases:
            task_id = client.submit(
                {
                    **case,
                    "mode": "xtb",
                    "optimizer_spec": optimizer_spec,
                }
            )
            submitted.append((task_id, case["mol_name"]))

        results: list[dict] = []
        num_errors = 0
        for task_id, mol_name in submitted:
            try:
                payload = client.wait_for_result(task_id, timeout_seconds=self.task_timeout_xtb)
            except (RuntimeError, TimeoutError) as exc:
                print(f"Error in {mol_name}: {exc}")
                num_errors += 1
                continue

            if payload["result"] is not None:
                results.append(payload["result"])
            if payload["error"] is not None:
                num_errors += 1

        if verbose:
            print(f"Done: {len(results)} results, {num_errors} errors")
        return {"results": results, "num_errors": num_errors}


def validate(
    optimizer: Callable | bytes | bytearray | memoryview | str | dict[str, Any],
    molecules_dir: str | os.PathLike[str] = REPO_ROOT / "molecules",
    split: str = "train",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    verbose: bool = False,
) -> dict:
    evaluator = Evaluator(
        molecules_dir=molecules_dir,
        split=split,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    start_time = time.time()
    result = evaluator.evaluate(optimizer, verbose=verbose)
    duration_s = time.time() - start_time
    score = score_results(result["results"], result["num_errors"])
    score["duration_s"] = duration_s
    score["num_results"] = len(result["results"])
    score["num_errors"] = result["num_errors"]

    line = (
        f"{datetime.now().isoformat(timespec='seconds')} | {duration_s:>7.1f}s | "
        f"fitness={score['fitness']:.6f} | mean_rel_steps={score['mean_rel_steps']:.6f} | "
        f"mean_rel_energy={score['mean_rel_energy']:.6f} | valid={score['is_valid']}"
    )
    if score["invalid_reason"]:
        line += f" | invalid_reason={score['invalid_reason']}"
    with open(REPO_ROOT / "validate_debug.log", "a", encoding="utf-8") as file_obj:
        file_obj.write(line + "\n")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the editable Sella optimizer.")
    parser.add_argument("--program", default="sella_tiny.py", help="Optimizer file to evaluate")
    parser.add_argument("--molecules-dir", default=str(REPO_ROOT / "molecules"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    optimizer_payload = serialize_program_entrypoint(args.program)
    result = validate(
        optimizer_payload,
        molecules_dir=args.molecules_dir,
        split=args.split,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        verbose=args.verbose,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
