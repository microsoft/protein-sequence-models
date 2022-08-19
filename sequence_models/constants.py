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
# PAD = GAP
PAD = '!'
MASK = '#'  # Useful for masked language model training
START = '@'

SPECIALS = STOP + GAP + MASK + START
PROTEIN_ALPHABET = ALL_AAS + SPECIALS + PAD
RNA_ALPHABET = IUPAC_AMB_RNA + SPECIALS

BLOSUM62_AAS = 'ARNDCQEGHILKMFPSTWYVBZX'  # In order of BLOSUM indices for matrix creation
ALL_AAS_BLOSUM = BLOSUM62_AAS + OTHER_AAS
PROTEIN_ALPHABET_BLOSUM = ALL_AAS + PAD + MASK

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

IUPAC_CODES = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Val": "V",
    "Tyr": "Y",
    "Asx": "B",
    "Sec": "U",
    "Xaa": "X",
    "Glx": "Z",
}
