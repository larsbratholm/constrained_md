"""
Generates cp2k input files for constrained md
"""
# Get dirpath of this script
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import openbabel as ob


def check_md_vars(dump_frequency, temperature, steps, timestep, charge, radical):
    assert dump_frequency > 0
    assert isinstance(dump_frequency, int)
    assert temperature > 0
    assert steps > 0
    assert isinstance(steps, int)
    assert timestep > 0
    assert dump_frequency < steps
    assert isinstance(charge, int)
    assert isinstance(radical, bool)

def parse_xyz_file(filename):
    """
    Parse an XYZ file and return atom types and carteesian coordinates
    """
    with open(filename) as f:
        lines = f.readlines()
    return parse_xyz(lines)

def parse_xyz_string(xyz):
    """
    Parse an XYZ string and return atom types and carteesian coordinates
    """
    lines = xyz.split("\n")[:-1]
    return parse_xyz(lines)

def parse_xyz(lines):
    coordinates = []
    atomtypes = []

    for line in lines[2:]:
        tokens = line.split()
        atomtypes.append(tokens[0])
        coordinates.append(tokens[1:])
    return np.asarray(atomtypes), np.asarray(coordinates, dtype=float)

def get_molpro_coordinates(atomtypes, coordinates):
    """
    Convert atomtypes and coordinates to molpro xyz format
    with named atom types
    """

    n = len(atomtypes)
    uniq, counts = np.unique(atomtypes, return_counts=True)

    xyz = "%d \n\n" % n
    c = 0
    for i in range(n):
        xyz += "%s%d %.6f %.6f %.6f\n" % (atomtypes[i], c, *coordinates[i])
        c += 1

    return xyz

def write_batch_md_input(xyz_filename, elements, basenames, constraints,
        **kwargs):
    """
    Batch version of write_md_input
    """
    for basename, constraint in zip(basenames, constraints):
        write_md_input(xyz_filename, elements, basename, constraint, **kwargs)

def write_batch_opt_input(xyz_filename, basenames, constraints,
        **kwargs):
    """
    Batch version of write_opt_input
    """
    for basename, constraint in zip(basenames, constraints):
        write_opt_input(xyz_filename, basename, constraint, **kwargs)
        quit()

def constrained_ff_optimization(xyz_filename, constraints):
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

    ## If only two constraints, then add additional angle constraint
    #if len(constraints) == 2:
    #    idx1 = constraints[0][0]
    #    idx2 = constraints[0][1]
    #    idx3 = constraints[1][0]
    #    idx4 = constraints[1][1]

    #    #if idx1 == idx3:
    #    #    ffconstraints.AddAngleConstraint(int(idx2), int(idx1), int(idx4), 180.0)
    #    #    #ffconstraints.AddAngleConstraint(int(idx2), int(idx1), int(idx4)+1, 180.0)
    #    #elif idx1 == idx4:
    #    #    ffconstraints.AddAngleConstraint(int(idx2), int(idx1), int(idx3), 180.0)
    #    #elif idx2 == idx3:
    #    #    ffconstraints.AddAngleConstraint(int(idx1), int(idx2), int(idx4), 180.0)
    #    #elif idx2 == idx4:
    #    #    ffconstraints.AddAngleConstraint(int(idx1), int(idx2), int(idx3), 180.0)




    # Setup the force field with the constraints
    forcefield = ob.OBForceField.FindForceField("UFF")
    forcefield.Setup(mol, ffconstraints)
    forcefield.SetConstraints(ffconstraints)

    # Do a 1000 steps conjugate gradient minimization
    # and save the coordinates to mol.
    forcefield.ConjugateGradients(500)

    forcefield.GetCoordinates(mol)

    return conv, mol

def write_md_input(xyz_filename, elements, basename, constraints,
        dump_frequency=100, temperature=300, steps=300, timestep=0.25,
        charge=0, radical=False, cell="ABC 20.0 20.0 20.0"):
    """
    Writes a cp2k inputfile in 'basename.inp'.
    'constraints' is a list of single constraints.
    Each single constraint is a tuple of the form
    (index_1, index_2, distance)
    """
    check_md_vars(dump_frequency, temperature, steps, timestep, charge, radical)

    # Read in the input template
    with open(dir_path + "/basemdinput.txt") as f:
        inp = f.read()

    # Change MD parameters
    inp = inp.replace("$var_name", basename)
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
    conv, mol = constrained_ff_optimization(xyz_filename, constraints)

    # Write the mol to a file
    conv.WriteFile(mol, outname)

def write_opt_input(xyz_filename, basename, constraints):
    """
    Writes a molpro inputfile in 'basename.com'.
    'constraints' is a list of single constraints.
    Each single constraint is a tuple of the form
    (index_1, index_2, distance)
    """

    # Read in the input template
    with open(dir_path + "/baseoptinput.txt") as f:
        inp = f.read()

    # Do a constrained MM FF optimization to get initial structure
    conv, mol = constrained_ff_optimization(xyz_filename, constraints)
    xyz_string = conv.WriteString(mol)
    conv.WriteFile(mol, basename + ".xyz")

    # Parse xyz
    atomtypes, coordinates = parse_xyz_string(xyz_string)

    # Change opt parameters
    inp = inp.replace("$var_xyzfilename",basename + ".xyz")

    con = "constraint,$var_distance,angstrom,bond,atoms=[$var_pos1,$var_pos2]\n"

    constraint_block = ""
    for index1, index2, distance in constraints:
        this_constraint = con.replace("$var_distance", "%.3f" % distance)
        this_constraint = this_constraint.replace("$var_pos1", str(atomtypes[index1]) + str(index1))
        this_constraint = this_constraint.replace("$var_pos2", str(atomtypes[index2]) + str(index2))
        constraint_block += this_constraint

    # Additional hardcoded angle constraint
    con = "constraint,180,deg,angle,atoms=[$var_pos1,$var_pos2,$var_pos3]\n"

    constraint_block = ""
    index1 = constraints[0][1]
    index2 = constraints[0][0]
    index3 = constraints[1][1]
    this_constraint = con.replace("$var_pos1", str(atomtypes[index1]) + str(index1))
    this_constraint = this_constraint.replace("$var_pos2", str(atomtypes[index2]) + str(index2))
    this_constraint = this_constraint.replace("$var_pos3", str(atomtypes[index3]) + str(index3))
    constraint_block += this_constraint

    inp = inp.replace("$block_constraints", constraint_block)

    # Write input file
    with open(basename + ".com", "w") as f:
        f.write(inp)

