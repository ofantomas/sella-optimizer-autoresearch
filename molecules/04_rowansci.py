import os
import sys
import json
import numpy as np
import openmm
from openmm import unit as omm_unit
from openff.toolkit import Molecule, ForceField
from rdkit import Chem
from xyz2mol import read_xyz_file, xyz2mol
from tqdm import tqdm

from importlib import import_module

# Make utils at repo root importable when running from molecules/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ANGSTROM_TO_NM, write_xyz

_orca = import_module("03_orca")

MOLECULES_DIR = _orca.MOLECULES_DIR
INPUT_XYZ_DIR = os.path.join(MOLECULES_DIR, "xyz_rowansci_benchmark")
OUTPUT_XYZ_DIR = os.path.join(MOLECULES_DIR, "xyz")
OUTPUT_JSON = "baseline_XTB_rowansci.json"
TOLERANCE = 10.0
MAX_ITERATIONS = 1000


def create_openmm_system_from_xyz(xyz_path):
    atomicNumList, charge, coords_ang = read_xyz_file(xyz_path)
    mol = xyz2mol(atomicNumList, coords_ang, charge=charge)[0]
    Chem.SanitizeMol(mol)

    off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    topology = off_mol.to_topology()
    ff = ForceField("openff_unconstrained-2.3.0.offxml")
    system = ff.create_openmm_system(topology)

    atomic_numbers = [at.atomic_number for at in topology.atoms]
    return (
        system,
        coords_ang * ANGSTROM_TO_NM,
        np.array(atomic_numbers, dtype=int),
    )


def minimize(system, positions_nm, tolerance, max_iterations):
    integrator = openmm.VerletIntegrator(0.001)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions_nm)

    state_before = context.getState(getEnergy=True)
    energy_before = state_before.getPotentialEnergy().value_in_unit(
        omm_unit.kilojoule_per_mole
    )

    openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)

    state_after = context.getState(getPositions=True, getEnergy=True)
    energy_after = state_after.getPotentialEnergy().value_in_unit(
        omm_unit.kilojoule_per_mole
    )
    final_positions = np.array(
        state_after.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer),
        dtype=np.float64,
    )
    del context
    del integrator

    return final_positions, energy_before, energy_after


def main():
    os.makedirs(OUTPUT_XYZ_DIR, exist_ok=True)

    xyz_files = sorted(f for f in os.listdir(INPUT_XYZ_DIR) if f.endswith(".xyz"))
    print(f"Found {len(xyz_files)} XYZ files in {INPUT_XYZ_DIR}")

    baselines = {}
    for filename in tqdm(xyz_files, desc="Rowansci benchmark"):
        name = os.path.splitext(filename)[0]
        xyz_path = os.path.join(INPUT_XYZ_DIR, filename)

        try:
            system, initial_positions, atomic_numbers = create_openmm_system_from_xyz(
                xyz_path
            )
            mm_positions, _, _ = minimize(
                system, initial_positions, TOLERANCE, MAX_ITERATIONS
            )

            out_xyz = os.path.join(OUTPUT_XYZ_DIR, f"{name}_mm.xyz")
            write_xyz(atomic_numbers, mm_positions, out_xyz)

            final_positions, initial_E, final_E, n_steps, converged, _ = (
                _orca.orca_optimize(out_xyz)
            )

            baselines[name] = {
                "n_atoms": len(atomic_numbers),
                "initial_energy": initial_E,
                "final_energy": final_E,
                "n_steps": n_steps,
                "converged": converged,
            }

        except Exception as e:
            print(f"Error {name}: {e}")

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
