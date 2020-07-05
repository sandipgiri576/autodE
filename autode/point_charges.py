import numpy as np


class PointCharge:

    def __init__(self, charge, x=0.0, y=0.0, z=0.0, coord=None):
        """
        Point charge

        Arguments:
            charge (any): Charge in units of e

        Keyword Arguments:
            x (any): x coordinate in
            y (any):
            z (any):
            coord (np.ndarray): Length 3 array of x, y, z coordinates or None
        """
        self.charge = float(charge)
        self.coord = np.array([float(x), float(y), float(z)])

        # If initialised with a coordinate override the default
        if coord is not None:
            assert type(coord) is np.ndarray
            assert len(coord) == 3
            self.coord = coord


def get_species_point_charges(species):
    """Gets a list of point charges for a species' mm_solvent_atoms
    List has the form: float of point charge, x, y, z coordinates"""
    if not hasattr(species, 'mm_solvent_atoms') or species.mm_solvent_atoms is None:
        return None

    point_charges = []
    for i, atom in enumerate(species.mm_solvent_atoms):
        charge = species.solvent_mol.graph.nodes[i % species.solvent_mol.n_atoms]['charge']
        point_charges.append(PointCharge(charge, coord=atom.coord))

    return point_charges
