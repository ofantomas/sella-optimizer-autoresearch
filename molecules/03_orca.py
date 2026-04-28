import os
import re
import sys
import json
import tempfile
import subprocess
import numpy as np
from tqdm import tqdm

# Make utils at repo root importable when running from molecules/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ANGSTROM_TO_NM, ATOM_ENERGIES_HARTREE, SYMBOL_TO_Z, parse_xyz


MOLECULES_DIR = "./"
CONF_INDICES_FILE = "conf_indices_1.json"
OUTPUT_JSON = "baseline_XTB.json"

ORCA_BIN = "/home/potapov/build/orca_6_0_0_shared_openmpi416/orca"
XTB_EXE = "/mnt/new_sfs_turbo/deshchenya/miniconda3/envs/openmm/bin/xtb"


def _write_orca_inp(path, xyz_path):
    with open(path, "w") as fh:
        fh.write("! XTB2 OPT\n")
        fh.write("%GEOM\n")
        fh.write("  EnforceStrictConvergence True\n")
        fh.write("END\n")
        fh.write(f"* xyzfile 0 1 {xyz_path}\n")


def _parse_orca_output(text, n_atoms):
    if isinstance(text, (str, os.PathLike)) and os.path.isfile(str(text)):
        with open(text, encoding="utf-8", errors="replace") as fh:
            text = fh.read()

    converged = "THE OPTIMIZATION HAS CONVERGED" in text

    initial_match = re.search(
        r"(?:FINAL SINGLE POINT ENERGY|Total Energy)\s*:?\s+([-0-9.eE]+)", text
    )
    initial_E_Eh = float(initial_match.group(1)) if initial_match else None

    final_matches = list(
        re.finditer(
            r"(?:FINAL SINGLE POINT ENERGY|Total Energy)\s*:?\s+([-0-9.eE]+)", text
        )
    )
    final_E_Eh = float(final_matches[-1].group(1)) if final_matches else None

    cycle_hdr = [
        int(m)
        for m in re.findall(
            r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", text, flags=re.IGNORECASE
        )
    ]
    n_cycles = max(cycle_hdr)
    max_cycles = max(50, 3 * n_atoms)

    if n_cycles > max_cycles:
        raise RuntimeError(
            f"Number of steps {n_cycles} exceeded maximum allowed ({max_cycles}) in ORCA optimization"
        )

    if not converged and n_cycles == max_cycles:
        print(f"Warning: not converged after {n_cycles} cycles (max={max_cycles})")
    else:
        after_match = re.search(r"AFTER\s+(\d+)\s+CYCLES", text, flags=re.IGNORECASE)
        if not after_match:
            raise RuntimeError(
                "Molecule optimization failed: 'AFTER ... CYCLES' line not found in ORCA output."
            )

        n_cycles_final = int(after_match.group(1))

        if n_cycles != n_cycles_final:
            raise RuntimeError(
                f"Cycle number mismatch: headers={n_cycles}, after={n_cycles_final}"
            )

    if initial_E_Eh is None or final_E_Eh is None:
        raise RuntimeError(
            f"Could not parse energies from ORCA output (initial={initial_E_Eh}, final={final_E_Eh})"
        )

    return initial_E_Eh, final_E_Eh, n_cycles, converged


def _parse_orca_xyz(path):
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0].strip())
    pos = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2 : 2 + n]])
    return pos * ANGSTROM_TO_NM


def orca_optimize(xyz_path):
    with open(xyz_path) as fh:
        lines = fh.readlines()
    n = int(lines[0].strip())
    symbols = [ln.split()[0] for ln in lines[2 : 2 + n]]
    ref_E = sum(ATOM_ENERGIES_HARTREE[SYMBOL_TO_Z[s]] for s in symbols)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_xyz = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path) as src, open(local_xyz, "w") as dst:
            dst.write(src.read())

        inp_path = os.path.join(tmpdir, "opt.inp")
        _write_orca_inp(inp_path, local_xyz)

        env = os.environ.copy()
        env["XTBEXE"] = XTB_EXE
        env["OMP_NUM_THREADS"] = "1"

        proc = subprocess.run(
            [ORCA_BIN, "opt.inp"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )

        if len(proc.stdout) < 100:
            raise RuntimeError(
                f"ORCA output missing; stdout: {proc.stdout[-500:]}, stderr: {proc.stderr[-500:]}"
            )

        initial_E_Eh, final_E_Eh, n_cycles, converged = _parse_orca_output(
            proc.stdout, n
        )
        initial_E = initial_E_Eh - ref_E
        final_E = final_E_Eh - ref_E

        opt_xyz = os.path.join(tmpdir, "opt.xyz")
        if os.path.exists(opt_xyz):
            final_positions = _parse_orca_xyz(opt_xyz)
        else:
            raise RuntimeError("opt.xyz missing; check ORCA output")

    return final_positions, initial_E, final_E, n_cycles, converged, proc.stdout


def main():
    conf_indices_path = os.path.join(MOLECULES_DIR, CONF_INDICES_FILE)
    with open(conf_indices_path, "r") as f:
        conf_indices = json.load(f)

    xyz_dir = os.path.join(MOLECULES_DIR, "xyz")

    baselines = {}
    for seed, conf_ids in tqdm(conf_indices.items(), desc="ORCA XTB2 optimization"):
        conf_id = conf_ids[0]

        xyz_path = os.path.join(xyz_dir, f"{seed}_{conf_id}_mm.xyz")
        atomic_numbers, _ = parse_xyz(xyz_path)

        try:
            final_positions, initial_E, final_E, n_steps, converged, _ = orca_optimize(
                xyz_path
            )
        except Exception as e:
            print(f"Error with molecule {seed}: {e}.")
            continue

        baselines[f"{seed}_{conf_id}"] = {
            "n_atoms": len(atomic_numbers),
            "initial_energy": initial_E,  # Hartree
            "final_energy": final_E,  # Hartree
            "n_steps": n_steps,
            "converged": converged,
        }

    output_path = os.path.join(MOLECULES_DIR, OUTPUT_JSON)
    with open(output_path, "w") as f:
        json.dump(
            baselines,
            f,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )
    print(f"Saved baselines for {len(baselines)} molecules -> {output_path}")


if __name__ == "__main__":
    main()
