"""
Generates cp2k input files for constrained md
"""
# Get dirpath of this script
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import openbabel as ob


def check_vars(dump_frequency, temperature, steps, timestep, charge, radical):
    assert dump_frequency > 0
    assert isinstance(dump_frequency, int)
    assert temperature > 0
    assert steps > 0
    assert isinstance(steps, int)
    assert timestep > 0
    assert dump_frequency < steps
    assert isinstance(charge, int)
    assert isinstance(radical, bool)

def parse_xyz(filename):
    """
    Parse an XYZ file and return atom types and carteesian coordinates
    """
    with open(filename) as f:
        lines = f.readlines()

    coordinates = []
    atomtypes = []

    for line in lines[2:]:
        tokens = line.split()
        atomtypes.append(tokens[0])
        coordinates.append(tokens[1:])
    return np.asarray(atomtypes), np.asarray(coordinates, dtype=float)

def write_batch_input(xyz_filename, elements, basenames, constraints,
        **kwargs):
    """
    Batch version of write_input
    """
    for basename, constraint in zip(basenames, constraints):
        write_input(xyz_filename, elements, basename, constraint, **kwargs)

def constrained_optimization(xyz_filename, outname, constraints):
    """
    Do a constrained optimization with UFF to make
    sure that the ab initio MD won't explode
    Code modified from gist.github.com/andersx/7784817
    """

    # Standard openbabel molecule load
    conv = ob.OBConversion()
    conv.SetInAndOutFormats('xyz','xyz')
    mol = ob.OBMol()
    conv.ReadFile(mol,xyz_filename)

    # Define constraints
    ffconstraints = ob.OBFFConstraints()
    for idx1, idx2, distance in constraints:
        ffconstraints.AddDistanceConstraint(int(idx1)+1, int(idx2)+1, float(distance))

    # Setup the force field with the constraints
    forcefield = ob.OBForceField.FindForceField("UFF")
    forcefield.Setup(mol, ffconstraints)
    forcefield.SetConstraints(ffconstraints)

    # Do a 500 steps conjugate gradient minimiazation
    # and save the coordinates to mol.
    forcefield.ConjugateGradients(500)
    forcefield.GetCoordinates(mol)

    # Write the mol to a file
    conv.WriteFile(mol, outname)

def write_input(xyz_filename, elements, basename, constraints,
        dump_frequency=100, temperature=300, steps=300, timestep=0.25,
        charge=0, radical=False, cell="ABC 20.0 20.0 20.0"):
    """
    Writes a cp2k inputfile in 'basename.inp'.
    'constraints' is a list of single constraints.
    Each single constraint is a tuple of the form
    (index_1, index_2, distance)
    """
    check_vars(dump_frequency, temperature, steps, timestep, charge, radical)

    # Read in the input template
    with open(dir_path + "/baseinput.txt") as f:
        inp = f.read()

    # Change MD parameters
    inp = inp.replace("$var_name", basename)
    inp = inp.replace("$var_dump_frequency", str(dump_frequency))
    inp = inp.replace("$var_dump_frequency", str(dump_frequency))
    inp = inp.replace("$var_temperature", str(temperature))
    inp = inp.replace("$var_steps", str(steps))
    inp = inp.replace("$var_timestep", str(timestep))
    inp = inp.replace("$var_cell", cell)
    outname = basename + ".xyz"
    inp = inp.replace("$var_filename", outname)


    # Add constraints
    # First the collective block
    with open(dir_path + "/baseconstraint1.txt") as f:
        con = f.read()

    constraint_block = ""
    c = 1
    for index1, index2, distance in constraints:
        this_constraint = con.replace("$var_constraint_n", str(c))
        this_constraint = this_constraint.replace("$var_constraint_distance", "%.2f" % distance)
        constraint_block += this_constraint
        c += 1

    inp = inp.replace("$block_constraints1", constraint_block)

    # Then the colvar block
    with open(dir_path + "/baseconstraint2.txt") as f:
        con = f.read()

    constraint_block = ""
    for index1, index2, distance in constraints:
        this_constraint = con.replace("$var_pos1", str(int(index1)+1))
        this_constraint = this_constraint.replace("$var_pos2", str(int(index2)+1))
        constraint_block += this_constraint

    inp = inp.replace("$block_constraints2", constraint_block)

    # Add basis set / potential for every atom type
    with open(dir_path + "/valence_electrons.txt") as f:
        lines = f.readlines()
    valence = {}
    for line in lines:
        tokens = line.split()
        valence[tokens[0]] = tokens[1]

    with open(dir_path + "/basepotential.txt") as f:
        pot = f.read()

    potential_block = ""
    for atype in elements:
        this_pot = pot.replace("$var_atomtype", atype)
        this_pot = this_pot.replace("$var_valence_electrons", valence[atype])
        potential_block += this_pot

    inp = inp.replace("$block_kind", potential_block)

    # Add charge and spin
    charge_block = ""
    if charge != 0:
        charge_block += "    CHARGE %d\n" % charge
    if radical:
        charge_block += "    LSD\n"

    inp = inp.replace("$block_charge", charge_block)

    # Write input file
    with open(basename + ".inp", "w") as f:
        f.write(inp)

    # Do a constrained MM FF optimization to get initial structure
    constrained_optimization(xyz_filename, outname, constraints)
