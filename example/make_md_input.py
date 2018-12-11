# Add generate_input script to path
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../src')))

from generate_input import write_batch_md_input, parse_xyz_file
import numpy as np

def get_constraints(atomtypes, coordinates):

    # Find all hydrogens
    h_idx = np.where(atomtypes == "H")[0]
    # Find all carbons
    c_idx = np.where(atomtypes == "C")[0]
    # Get all distances between carbons and hydrogens
    ch_distances = np.sqrt(np.sum((coordinates[h_idx, None] - coordinates[None, c_idx])**2, axis=2))
    # Classify bonds as where C-H distance is less than 1.5 Å
    idx1, idx2 = np.where(ch_distances < 1.5)
    # Get indices of H-C bonded pairs
    hydrogens = h_idx[idx1]
    carbons = c_idx[idx2]

    # Cyanide C index
    cyanide_c = 17

    # Basic range for distances
    distance_constraints = [0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]

    # constraints of the form (atom_index_1, atom_index_2, distance), using python indexing
    constraints = []
    basenames = []
    for dist1 in distance_constraints:
        for dist2 in distance_constraints:
            # Ignore the ones where H is completely non-bonded
            if dist1 > 2.51 and dist2 > 2.51:
                continue
            for idxh, idxc in zip(hydrogens, carbons):
                constraints.append([(idxh, idxc, dist1),(idxh, cyanide_c, dist2)])
                basename = "%d_%.2f_%.2f" % (idxh, dist1, dist2)
                basenames.append(basename)

    return constraints, basenames

if __name__ == "__main__":

    filename = "isopentane_cn.xyz"

    atomtypes, coordinates = parse_xyz_file(filename)
    constraints, basenames = get_constraints(atomtypes, coordinates)
    elements = np.unique(atomtypes)

    write_batch_md_input(filename, elements, basenames, constraints,
        dump_frequency=50, temperature=300, steps=1000, timestep=0.25,
        charge=0, radical=True)

