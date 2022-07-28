import os, sys
import numpy as np

from sequence_models import pdb_utils

ex_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)
assert os.path.isdir(ex_dir)

orig_coords, orig_atoms, orig_valid = pdb_utils.parse_PDB(
    os.path.join(ex_dir, "gb1_a60fb_unrelaxed_rank_1_model_5.pdb")
)
gz_coords, gz_atoms, gz_valid = pdb_utils.parse_PDB(
    os.path.join(ex_dir, "gb1_a60fb_unrelaxed_rank_1_model_5.pdb.gz")
)

assert np.all(np.isclose(orig_coords, gz_coords))
assert orig_atoms == gz_atoms
assert np.all(orig_valid == gz_valid)
