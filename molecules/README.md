# Molecule Benchmark Pipeline

Pipeline for preparing molecules (from the SPICE 2.0.1 dataset and the Rowansci drug benchmark), minimizing with OpenMM, and producing XTB2 baselines via ORCA.

## Scripts

Run in order:

```
python 01_prepare.py
python 02_openmm.py
python 03_orca.py
python 04_rowansci.py
python 05_train_test_split.py
```

### `01_prepare.py`

Downloads `sorted.txt` (molecule list from SPICE dataset) and `SPICE-2.0.1.hdf5` from Zenodo. Both are skipped if already present. Loads up to 1500 molecules, attempts OpenFF parameterization (`openff_unconstrained-2.3.0`), and stops after 1000 successfully parameterized systems. Skips molecules whose files already exist (allows restart).

For each successful molecule, saves three files to `openmm/`:

- `mol_{seed}_confs.npy` — conformations in nm (shape: `n_confs x n_atoms x 3`)
- `mol_{seed}_system.xml` — serialized OpenMM system (forces, constraints)
- `mol_{seed}_numbers.npy` — atomic numbers (length: `n_atoms`)

`**log.txt**` — tab-separated log of all successfully parameterized molecules. Columns: `seed`, `n_atoms`, `n_heavy_atoms`, `n_confs`, `smiles`, `mapped_smiles`. Lines starting with `#` mark failures. This file is read by `create_conf_indices` to build the conformation selection.

`**conf_indices_1.json**` — maps each molecule seed (string) to a list of selected conformation indices (length 1). For molecules with fewer conformations than requested, all conformations are included. Random selection uses seed 42.

**Outputs:** `openmm/`, `log.txt`, `conf_indices_1.json`

### `02_openmm.py`

Reads `conf_indices_1.json` and corresponding files from `openmm/`. For each molecule, loads the OpenMM system, sets initial positions from the selected conformation, and minimizes using `LocalEnergyMinimizer` (Reference platform, tolerance=10 kJ/(molnm), max 1000 iterations). Saves the optimized geometry as an XYZ file with element symbols.

**Output naming:** `xyz/{seed}_{conf_id}_mm.xyz` (e.g., `103927616_23_mm.xyz`)

### `03_orca.py`

Reads `conf_indices_1.json` and XYZ files from `xyz/`. For each molecule, writes an ORCA input with `! XTB2 OPT`, copies the XYZ to a temp directory, and runs ORCA. Parses `FINAL SINGLE POINT ENERGY` lines for initial/final energies, counts optimization cycles from `GEOMETRY OPTIMIZATION CYCLE` headers, and checks convergence via `THE OPTIMIZATION HAS CONVERGED`. Molecules where SCF fails (e.g., zero HOMO-LUMO gap) are skipped.

Energies are in Hartree, relative to isolated atom references.

**Output format** (`baseline_XTB.json`):

```json
{
  "{seed}_{conf_id}": {
    "n_atoms": 26,
    "initial_energy": -0.042,    // Hartree
    "final_energy": -0.051,      // Hartree
    "n_steps": 11,
    "converged": true
  }
}
```

Keys follow `{seed}_{conf_id}` so they match the `xyz/{seed}_{conf_id}_mm.xyz` filenames (XYZ is the source of truth for geometry; the JSON stores scalar metrics only).

### `04_rowansci.py`

Processes drug molecule XYZ files from `xyz_rowansci_benchmark/` (25 molecules). For each:

1. Infers bonds and bond orders from XYZ coordinates using `xyz2mol` (no external SDF or topology files needed)
2. Creates OpenFF molecule from the RDKit result
3. Parameterizes with OpenFF, creates OpenMM system (in memory, no files saved)
4. Minimizes with OpenMM, writes optimized XYZ to `xyz/{name}_mm.xyz`
5. Runs ORCA XTB2 optimization on the optimized XYZ
6. Saves baseline to `baseline_XTB_rowansci.json` (same schema as `baseline_XTB.json`, but keys are molecule names like `"abemaciclib"` that match `xyz/{name}_mm.xyz`)

Imports constants and `orca_optimize` from `03_orca.py` via `importlib.import_module`.

### `05_train_test_split.py`

Reads `baseline_XTB.json` (999 SPICE molecules, keyed `{seed}_{conf_id}`) and `baseline_XTB_rowansci.json` (25 drug molecules, keyed by name). Creates two files:

- `**train_XTB.json**` — first 225 molecules from `baseline_XTB.json` merged with all 25 molecules from `baseline_XTB_rowansci.json` (250 total).
- `**test_XTB.json**` — 250 molecules from `baseline_XTB.json` starting after the first 225 (indices 225–474).

Both files use the same JSON schema as the baselines. Each key maps directly to an XYZ file under `xyz/{key}_mm.xyz` for uniform geometry lookup across SPICE and Rowansci entries.

**Filtering:** `IMPROVEMENT_MIN` (Hartree, default `1e-4` ≈ 0.26 kJ/mol) drops entries whose baseline improvement (`initial_energy - final_energy`) is below the threshold. Applied to both XTB and Rowansci baselines before the train/test split. Guards against degenerate `rel_energy` values in validation where the baseline already reached a near-optimal geometry (tiny denominator → blowup).

## Directory Structure

```
molecules/
  SPICE-2.0.1.hdf5           # SPICE dataset (auto-downloaded, ~3GB)
  sorted.txt                 # sorted molecule list (auto-downloaded)
  log.txt                    # parameterized molecule log (tab-separated)
  conf_indices_1.json        # seed -> [conf_id] mapping
  baseline_XTB.json          # ORCA baselines for SPICE molecules
  baseline_XTB_rowansci.json # ORCA baselines for Rowansci drug molecules
  train_XTB.json             # training set (225 XTB + 25 Rowansci molecules)
  test_XTB.json              # test set (250 XTB molecules)
  openmm/                    # OpenMM system files (~1000 molecules)
    mol_{seed}_confs.npy     # conformations in nm
    mol_{seed}_system.xml    # serialized OpenMM system
    mol_{seed}_numbers.npy   # atomic numbers
  xyz/                       # optimized XYZ files from OpenMM (1000 files)
  xyz_rowansci_benchmark/    # input drug molecule XYZ files (25 files)
```

## Dependencies

- OpenMM
- OpenFF Toolkit (`openff_unconstrained-2.3.0.offxml`)
- RDKit
- xyz2mol
- h5py, numpy, pandas, requests, tqdm
- scipy
- ORCA 6.0 (with XTB2 via `XTBEXE` env variable)

