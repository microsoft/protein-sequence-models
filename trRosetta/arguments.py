import argparse

def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ALN", type=str, help="input multiple sequence alignment in A3M/FASTA format")
    parser.add_argument("NPZ", type=str, help="predicted distograms and anglegrams")
    parser.add_argument('-m', type=str, required=True, dest='MDIR', help='folder with the pre-trained network')

    args = parser.parse_args()

    return args
