import json

IMPROVEMENT_MIN = 1e-2  # Hartree; skip molecules with baseline (initial - final) < this


def _filter_by_improvement(baselines: dict, threshold: float) -> dict:
    kept = {}
    for key, entry in baselines.items():
        improvement = entry["initial_energy"] - entry["final_energy"]
        if improvement >= threshold:
            kept[key] = entry
    return kept


with open("baseline_XTB.json") as f:
    baseline_xtb = json.load(f)

with open("baseline_XTB_rowansci.json") as f:
    baseline_rowansci = json.load(f)

n_xtb_raw = len(baseline_xtb)
n_rowansci_raw = len(baseline_rowansci)
baseline_xtb = _filter_by_improvement(baseline_xtb, IMPROVEMENT_MIN)
baseline_rowansci = _filter_by_improvement(baseline_rowansci, IMPROVEMENT_MIN)
print(
    f"Filtered improvement < {IMPROVEMENT_MIN} Ha: "
    f"XTB {n_xtb_raw} -> {len(baseline_xtb)}, "
    f"Rowansci {n_rowansci_raw} -> {len(baseline_rowansci)}"
)

keys = list(baseline_xtb.keys())

train = {k: baseline_xtb[k] for k in keys[:225]}
train.update(baseline_rowansci)

test = {k: baseline_xtb[k] for k in keys[225:475]}

with open("train_XTB.json", "w") as f:
    json.dump(train, f)

with open("test_XTB.json", "w") as f:
    json.dump(test, f)

print(
    f"train_XTB.json: {len(train)} molecules ({len(train) - len(baseline_rowansci)} from XTB + {len(baseline_rowansci)} from Rowansci)"
)
print(f"test_XTB.json: {len(test)} molecules")
