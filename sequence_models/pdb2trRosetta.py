import os 
import pandas as pd
import numpy as np
import json
import json_lines
import scipy
from scipy import spatial
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse


parser = argparse.ArgumentParser(description='Convert PDB coords to trRosetta outputs')
parser.add_argument('--data_path', type=str, default='data/', help='path to jsonl with data for all proteins')
parser.add_argument('--split_path', type=str, default='', help='path to json with defined splits')
parser.add_argument('--out_path', type=str, default='', help='path to directory to save output')
parser.add_argument('--dmax', type=int, default=20, help='max distance between residue to consider to be in contact')
parser.add_argument('--n_jobs', type=int, default=-1, help='number of cores to use')

args = parser.parse_args()

def parse_CATH(path):
    """
    Converts data stored in jsonl into dictionary containing coords and seq info
    """
    coords_dict = {}
    with open(path, 'r') as f:
        for item in json_lines.reader(f):
            coords_dict[item['name']] = (item['coords'], item['seq'])
    return coords_dict

def parse_splits(path):
    """
    Extract data splits from json file, ignore everything else in json
    """
    with open(path) as json_file:
        splits = json.load(json_file)
        return {'train': splits['train'], 'test' : splits['test'], 'validation' : splits['validation']}
    
def get_dihedrals(a, b, c, d):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)


def coords_to_trRosetta(save_path, coords, seq, dmax):

#     nres = pyrosetta.rosetta.core.pose.nres_protein(pose)

    # three anchor atoms
#     N = np.stack([np.array(pose.residue(i).atom('N').xyz()) for i in range(1,nres+1)])
#     Ca = np.stack([np.array(pose.residue(i).atom('CA').xyz()) for i in range(1,nres+1)])
#     C = np.stack([np.array(pose.residue(i).atom('C').xyz()) for i in range(1,nres+1)])


    N = np.array(coords['N'])
    Ca = np.array(coords['CA']) 
    C = np.array(coords['C'])

    # recreate Cb given N,Ca,C
    nres = len(N)
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist = np.zeros((nres, nres))
    dist[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega = np.zeros((nres, nres))
    omega[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta = np.zeros((nres, nres))
    theta[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi = np.zeros((nres, nres))
    phi[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    np.savez(save_path, 
        dist = dist,
        omega = omega, 
        theta = theta,
        phi = phi,
        seq = np.array(seq))

    
# make directories based on splits
if not os.path.exists(args.out_path + '/train'):
    os.mkdir(args.out_path + '/train')
    
if not os.path.exists(args.out_path + '/test'):
    os.mkdir(args.out_path + '/test')

if not os.path.exists(args.out_path + '/validation'):
    os.mkdir(args.out_path + '/validation')
    
# load files
split_dict = parse_splits(args.split_path)
data_dict = parse_CATH(args.data_path)
    
# convert and save
# each npz file contains: dist, omega, theta, phi, and seq
for setname, setdata in split_dict.items():
    print('DATASET: ', setname)
    Parallel(n_jobs=args.n_jobs)(delayed(coords_to_trRosetta)( \
            save_path=args.out_path + '/' + setname + '/' + setdata[i] + '.npz', 
            coords=data_dict[setdata[i]][0], 
            seq=data_dict[setdata[i]][1],
            dmax=args.dmax) for i in tqdm(range(len(setdata))))
    