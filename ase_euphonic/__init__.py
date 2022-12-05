import ase
from ase.vibrations.data import VibrationsData
from euphonic import Crystal, ForceConstants, ureg
import numpy as np

def atoms_to_crystal(atoms: ase.Atoms) -> Crystal:
    return Crystal(atoms.cell.array * ureg('angstrom'),
                   atoms.get_scaled_positions(),
                   np.array(atoms.get_chemical_symbols(), dtype=str),
                   np.array(atoms.get_masses()) * ureg('amu'))

def ase_to_euphonic(vib: VibrationsData) -> ForceConstants:
    """Convert ASE vibrations to Euphonic ForceConstants

    Note that ASE vibrations calculations do not have a supercell definition
    (there is separate Phonons machinery for this) so the resulting
    ForceConstants is always defined with a (1, 1, 1) supercell.
    """

    crystal = atoms_to_crystal(vib.get_atoms())

    fc = vib.get_hessian_2d()[None, :, :] * ureg('eV / angstrom / angstrom')
    
    return ForceConstants(crystal, fc, np.eye(3),
                          np.array([[0, 0, 0]], dtype=int),
                          born=None, dielectric=None)
