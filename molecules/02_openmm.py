import os
import json
import numpy as np
import openmm
from openmm import unit as omm_unit
from tqdm import tqdm


MOLECULES_DIR = "./"
CONF_INDICES_FILE = "conf_indices_1.json"
TOLERANCE = 10.0
MAX_ITERATIONS = 1000

ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H",
    3: "Li",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    12: "Mg",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    20: "Ca",
    35: "Br",
    53: "I",
}


def write_xyz(atomic_numbers, positions_nm, output_path):
    pos_ang = positions_nm * 10.0
    n_atoms = len(atomic_numbers)
    with open(output_path, "w") as fh:
        fh.write(f"{n_atoms}\n\n")
        for number, (x, y, z) in zip(atomic_numbers, pos_ang):
            symbol = ATOMIC_NUMBER_TO_SYMBOL[number]
            fh.write(f"{symbol:<2} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def minimize(system, positions_nm, tolerance, max_iterations):
    integrator = openmm.VerletIntegrator(0.001)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions_nm)

    state_before = context.getState(getEnergy=True)
    energy_before = state_before.getPotentialEnergy().value_in_unit(
        omm_unit.kilojoule_per_mole
    )

    ret = openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)

    state_after = context.getState(getPositions=True, getEnergy=True)
    energy_after = state_after.getPotentialEnergy().value_in_unit(
        omm_unit.kilojoule_per_mole
    )
    final_positions = state_after.getPositions(asNumpy=True).value_in_unit(
        omm_unit.nanometer
    )
    # print(energy_before, energy_after)

    del context
    del integrator

    return np.array(final_positions, dtype=np.float64)


def main():
    openmm_dir = os.path.join(MOLECULES_DIR, "openmm")
    xyz_dir = os.path.join(MOLECULES_DIR, "xyz")
    os.makedirs(xyz_dir, exist_ok=True)

    conf_indices_path = os.path.join(MOLECULES_DIR, CONF_INDICES_FILE)
    with open(conf_indices_path, "r") as f:
        conf_indices = json.load(f)

    for seed, conf_ids in tqdm(conf_indices.items(), desc="OpenMM minimization"):
        try:
            conf_id = conf_ids[0]

            system_xml_path = os.path.join(openmm_dir, f"mol_{seed}_system.xml")
            with open(system_xml_path, "r") as f:
                system = openmm.XmlSerializer.deserialize(f.read())

            confs = np.load(os.path.join(openmm_dir, f"mol_{seed}_confs.npy"))
            initial_positions = confs[conf_id]

            atomic_numbers = np.load(
                os.path.join(openmm_dir, f"mol_{seed}_numbers.npy")
            )

            final_positions = minimize(
                system, initial_positions, TOLERANCE, MAX_ITERATIONS
            )

            write_xyz(
                atomic_numbers,
                final_positions,
                os.path.join(xyz_dir, f"{seed}_{conf_id}_mm.xyz"),
            )

        except Exception as e:
            print(f"Error {seed}: {e}")

    print(f"Done! Saved XYZ files to {xyz_dir}")


if __name__ == "__main__":
    main()
