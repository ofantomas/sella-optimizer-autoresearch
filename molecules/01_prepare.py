import os
import sys
import json
import numpy as np
import h5py
import requests
from tqdm import tqdm
import openmm
from openff.toolkit import Molecule, ForceField

# Make utils at repo root importable when running from molecules/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import BOHR_PER_NM


SPICE_HDF5_URL = "https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5"
SORTED_TXT_URL = "https://raw.githubusercontent.com/openmm/spice-dataset/refs/heads/main/pubchem/sorted.txt"
OUTPUT_DIR = "./"

N_CONFS_PER_MOL = 1
N_PARAMETERIZED = 1000
BOHR_TO_NM = 1.0 / BOHR_PER_NM


def download_sorted_txt(url, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
        return
    print(f"Downloading sorted.txt from {url}...")
    resp = requests.get(url)
    resp.raise_for_status()
    with open(output_path, "w") as f:
        f.write(resp.text)
    print(f"Saved to {output_path}")


def download_spice_hdf5(url, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
        return
    print(f"Downloading SPICE-2.0.1.hdf5 from {url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {output_path}")


def load_spice_systems(sorted_txt_path, hdf5_path, max_systems):
    spice_sorted = np.loadtxt(sorted_txt_path, dtype=str, comments=None)
    systems = []
    with h5py.File(hdf5_path, "r") as f:
        for seed, smiles in tqdm(spice_sorted, desc="Loading SPICE systems"):
            try:
                mapped_smiles = f[seed]["smiles"][0].decode()
                systems.append(
                    {"seed": seed, "smiles": smiles, "mapped_smiles": mapped_smiles}
                )
                if len(systems) == max_systems:
                    break
            except:
                pass

    return systems


def create_openff_systems(systems, output_dir, hdf5_path, max_parameterized):
    os.makedirs(output_dir, exist_ok=True)
    openmm_dir = os.path.join(output_dir, "openmm")
    os.makedirs(openmm_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    flog = open(log_path, "a")
    if flog.tell() == 0:
        flog.write("seed\tn_atoms\tn_heavy_atoms\tn_confs\tsmiles\tmapped_smiles\n")

    n_success = 0
    with h5py.File(hdf5_path, "r") as f:
        for system in tqdm(systems, desc="Creating OpenFF systems"):
            try:
                seed = str(system["seed"])
                smiles = system["smiles"]
                mapped_smiles = system["mapped_smiles"]

                confs_path = os.path.join(openmm_dir, f"mol_{seed}_confs.npy")
                system_path = os.path.join(openmm_dir, f"mol_{seed}_system.xml")
                numbers_path = os.path.join(openmm_dir, f"mol_{seed}_numbers.npy")

                if (
                    os.path.exists(confs_path)
                    and os.path.exists(system_path)
                    and os.path.exists(numbers_path)
                ):
                    n_success += 1
                    if n_success >= max_parameterized:
                        break
                    continue

                g = f[seed]

                conformations = (
                    np.asarray(g["conformations"][:], dtype=np.float64) * BOHR_TO_NM
                )
                n_confs = conformations.shape[0]

                molecule = Molecule.from_mapped_smiles(
                    mapped_smiles, allow_undefined_stereo=True
                )
                topology = molecule.to_topology()
                n_atoms = topology.n_atoms
                n_heavy_atoms = sum([at.atomic_number != 1 for at in topology.atoms])

                forcefield = ForceField("openff_unconstrained-2.3.0.offxml")
                openmm_system = forcefield.create_openmm_system(topology)

                np.save(confs_path, conformations)

                with open(system_path, "w") as fout:
                    fout.write(openmm.XmlSerializer.serialize(openmm_system))

                np.save(
                    numbers_path,
                    np.array([at.atomic_number for at in topology.atoms], dtype=int),
                )

                flog.write(
                    f"{seed}\t{n_atoms}\t{n_heavy_atoms}\t{n_confs}\t{smiles}\t{mapped_smiles}\n"
                )
                flog.flush()
                os.fsync(flog.fileno())

                n_success += 1
                if n_success >= max_parameterized:
                    break

            except Exception as e:
                flog.write(f"# {system['seed']}\terror\n")
                print(f"Error with {system['seed']}: {e}")

    flog.close()
    print(f"Done! Saved {n_success} systems to {openmm_dir}")


def create_conf_indices(output_dir, n_confs_per_mol):
    log_path = os.path.join(output_dir, "log.txt")
    import pandas as pd

    df = pd.read_csv(log_path, dtype=str, sep="\t")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    np.random.seed(42)
    conf_indices = {}

    for it, row in df.iterrows():
        seed = row.seed
        n_confs = int(row.n_confs)

        if n_confs < n_confs_per_mol:
            conf_indices[seed] = np.arange(n_confs).tolist()
        else:
            conf_indices[seed] = np.random.choice(
                n_confs, n_confs_per_mol, replace=False
            ).tolist()

    conf_indices_path = os.path.join(output_dir, f"conf_indices_{n_confs_per_mol}.json")
    with open(conf_indices_path, "w") as f:
        json.dump(conf_indices, f)
    print(
        f"Created conf_indices with {len(conf_indices)} molecules -> {conf_indices_path}"
    )


def main():
    sorted_txt_path = os.path.join(OUTPUT_DIR, "sorted.txt")
    spice_hdf5_path = os.path.join(OUTPUT_DIR, "SPICE-2.0.1.hdf5")

    download_sorted_txt(SORTED_TXT_URL, sorted_txt_path)
    download_spice_hdf5(SPICE_HDF5_URL, spice_hdf5_path)

    systems = load_spice_systems(
        sorted_txt_path, spice_hdf5_path, int(N_PARAMETERIZED * 1.5)
    )  # some will fail openff parameterization
    print(f"Loaded {len(systems)} systems from SPICE")

    create_openff_systems(systems, OUTPUT_DIR, spice_hdf5_path, N_PARAMETERIZED)

    create_conf_indices(OUTPUT_DIR, N_CONFS_PER_MOL)


if __name__ == "__main__":
    main()
