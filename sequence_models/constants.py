import pathlib

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

IUPAC_SS = 'HSTC'
DSSP = 'GHITEBSC'
SS8 = DSSP
SS3 = 'HSL'  # H: GHI; S: EB; L: STC


