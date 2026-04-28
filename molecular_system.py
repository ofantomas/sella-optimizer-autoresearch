"""
OpenMM-based molecular system for energy and force calculations.

Loads a serialized OpenMM system from an XML file. Atomic numbers and
starting positions should be obtained separately (e.g. via utils.parse_xyz)
and passed to compute().
"""

import numpy as np
import openmm
from openmm import unit as omm_unit


class MolecularSystem:
    """
    OpenMM-based molecular system for energy and force calculations.

    Args:
        system_xml_path: Path to a serialized OpenMM system XML file.
    """

    def __init__(self, system_xml_path: str):
        with open(system_xml_path, "r") as f:
            system = openmm.XmlSerializer.deserialize(f.read())

        integrator = openmm.VerletIntegrator(0.001)
        platform = openmm.Platform.getPlatformByName("Reference")
        self._context = openmm.Context(system, integrator, platform)

    def compute(self, positions, **kwargs) -> tuple[float, np.ndarray]:
        """
        Compute energy and forces.

        Args:
            positions: array-like (N, 3) in nm.

        Returns:
            (energy, forces): energy in kJ/mol, forces in kJ/(mol·nm).
        """
        positions = np.asarray(positions, dtype=float)
        try:
            self._context.setPositions(positions)
            state = self._context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                omm_unit.kilojoule_per_mole / omm_unit.nanometer
            )
            return float(energy), np.array(forces, dtype=float)
        except Exception:
            return float("inf"), np.zeros_like(positions, dtype=float)
