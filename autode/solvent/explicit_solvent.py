from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import os
from autode.atoms import get_vdw_radius
from autode.log import logger
from itertools import product
from autode.calculation import Calculation


def add_solvent_molecules(species):
    """Add a solvent molecules around a molceule's VdW surface"""
    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()

    # magic number for the distance of solvent from the molecule
    radius_mult = 0.78

    # magic number for the distance of solvent molecules from each other
    solv_mult = 1.248

    logger.info(f'Adding solvent molecules around {species.name}')

    centre_species(species.solvent_mol)
    solvent_coords = species.solvent_mol.get_coordinates()
    solvent_size = np.linalg.norm(np.max(solvent_coords, axis=0) - np.min(solvent_coords, axis=0))
    solvent_size = 1 if solvent_size < 1 else solvent_size

    points_on_surface = []

    thetas = np.arange(0, np.pi, 0.1)
    phis = np.arange(0, 2*np.pi, 0.1)

    for atom in species.atoms:
        # get points on a sphere around each atom
        r = get_vdw_radius(atom.label) + radius_mult * solvent_size
        x0, y0, z0 = atom.coord
        for theta, phi in product(thetas, phis):
            x = x0 + r * np.sin(theta) * np.cos(phi)
            y = y0 + r * np.sin(theta) * np.sin(phi)
            z = z0 + r * np.cos(theta)
            good = True
            for other_atom in species.atoms:
                # ensure the points aren't inside another atom
                other_r = get_vdw_radius(other_atom.label) + radius_mult * solvent_size
                other_x0, other_y0, other_z0 = other_atom.coord
                if (other_x0 - x)**2 + (other_y0 - y)**2 + (other_z0 - z)**2 < other_r**2:
                    good = False
                    break
            if good:
                points_on_surface.append(np.array([x, y, z]))

    good_solvent_points = []

    for point in points_on_surface:
        # discard any points too close to each other
        close = False
        for other_point in good_solvent_points:
            if np.linalg.norm(point-other_point) < solv_mult * solvent_size:
                close = True
                break
        if not close:
            good_solvent_points.append(point)

    solvent_atoms = []

    for point in good_solvent_points:
        # place a solvent molecule on each point
        species.solvent_mol.rotate(axis=rand.uniform(-1.0, 1.0, 3), theta=2*np.pi*rand.rand())
        for atom in species.solvent_mol.atoms:
            new_atom = deepcopy(atom)
            new_atom.translate(point)
            solvent_atoms.append(new_atom)

    species.solvent_atoms = solvent_atoms

    return None


def centre_species(species):
    """Translates a species so its centre is at (0,0,0)"""
    species_coords = species.get_coordinates()
    species_centre = np.average(species_coords, axis=0)
    species.translate(-species_centre)


def run_explicit_solvent_calcs(species, name, method, i):
    file_prefix = f'{name}_solv_{i}'
    add_solvent_molecules(species)

    species.atoms += species.solvent_atoms
    opt_calc = Calculation(f'{file_prefix}_opt', species, method, method.keywords.opt,
                           n_cores=1, cartesian_constraints=[i for i in range(species.n_atoms)])
    opt_calc.run()
    opt_atoms = opt_calc.get_final_atoms()

    species.atoms = opt_atoms
    full_sp = Calculation(f'{file_prefix}_full_sp', species, method,
                          method.keywords.sp, n_cores=1)
    full_sp.run()

    species.atoms = opt_atoms[species.n_atoms:]
    solvent_sp = Calculation(f'{file_prefix}_solvent_sp', species, method,
                             method.keywords.sp, n_cores=1)
    solvent_sp.run()

    energy = full_sp.get_energy() - solvent_sp.get_energy()

    return opt_atoms, energy


def do_explicit_solvent_calcs(species, name, method, n_cores, n_confs=50):
    """Run explicit solvent calculations to find the lowest energy of the solvated species"""

    logger.info(f'Running Explicit Solvent Calculation for {species.name}')

    logger.info(f'Splitting calculation into {n_cores} threads')
    with Pool(processes=n_cores) as pool:
        results = [pool.apply_async(run_explicit_solvent_calcs, (species, name, method, i)) for i in range(n_confs)]
        atoms_and_energies = [res.get(timeout=None) for res in results]

    explicit_atoms = []
    energies = []

    for atoms, energy in atoms_and_energies:
        explicit_atoms.append(atoms)
        energies.append(energy)

    min_e = min(energies)
    lowest_energy_atoms = explicit_atoms[energies.index(min_e)]

    # get a bolztmann weighting of the energy
    q = 0
    boltzmann_energy = 0
    for e in energies:
        energy = e - min_e
        q += np.exp(-1052.58*energy)
        boltzmann_energy += energy * np.exp(-1052.58*energy)
    boltzmann_energy = (boltzmann_energy / q) + min_e

    species_atoms = lowest_energy_atoms[:species.n_atoms]
    solvent_atoms = lowest_energy_atoms[species.n_atoms:]

    return boltzmann_energy, species_atoms, solvent_atoms
