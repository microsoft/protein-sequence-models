import pathlib
import os

import numpy as np

home = os.getenv('PT_DATA_DIR')
if home is None:
    home = str(pathlib.Path.home())
WEIGHTS_DIR = home + '/sm_weights/'

#  It's helpful to separate out the twenty canonical amino acids from the rest
CAN_AAS = 'ACDEFGHIKLMNPQRSTVWY'
AMB_AAS = 'BZX'
OTHER_AAS = 'JOU'
ALL_AAS = CAN_AAS + AMB_AAS + OTHER_AAS

DNA = 'GATC'
EXTENDED_NA = 'RYWSMKHBVDN'
RNA = 'GAUC'
IUPAC_AMB_DNA = DNA + EXTENDED_NA
IUPAC_AMB_RNA = RNA + EXTENDED_NA
NAS = 'GATUC' + EXTENDED_NA

STOP = '*'
GAP = '-'
PAD = GAP
MASK = '#'  # Useful for masked language model training
START = '@'

SPECIALS = STOP + GAP + MASK + START
PROTEIN_ALPHABET = ALL_AAS + SPECIALS
RNA_ALPHABET = IUPAC_AMB_RNA + SPECIALS

trR_ALPHABET = "ARNDCQEGHILKMFPSTWYV-"

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'

IUPAC_SS = 'HSTC'
DSSP = 'GHITEBSC'
SS8 = DSSP
SS3 = 'HSL'  # H: GHI; S: EB; L: STC

# Bins from TrRosetta paper
DIST_BINS = np.concatenate([np.array([np.nan]), np.linspace(2, 20, 37)])
THETA_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, 2 * np.pi, 25)])
PHI_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, np.pi, 13)])
OMEGA_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, 2 * np.pi, 25)])


