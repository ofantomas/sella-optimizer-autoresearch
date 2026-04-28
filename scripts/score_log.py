#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_last_json(path: Path) -> dict:
    found = None
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                found = json.loads(line)
    if found is None:
        raise SystemExit(f"No JSON result found in {path}")
    return found


def main() -> None:
    log_path = Path(sys.argv[1] if len(sys.argv) > 1 else "run.log")
    result = load_last_json(log_path)
    print(json.dumps(result, sort_keys=True))
    print(
        "fitness={fitness:.6f} mean_rel_steps={mean_rel_steps:.6f} "
        "mean_rel_energy={mean_rel_energy:.6f} is_valid={is_valid}".format(**result)
    )


if __name__ == "__main__":
    main()
