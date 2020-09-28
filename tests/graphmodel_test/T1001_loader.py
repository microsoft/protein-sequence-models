import os
import numpy as np
import string
from typing import Iterable

def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    return seqs

def logits2value(logits, bins):
    preds = np.argmax(logits, axis=2)
    retval = np.zeros(preds.shape)
    for i in range(len(preds)):
        for j in range(len(preds)):
            retval[i,j] = bins[preds[i,j]]
            
    return retval

def loadT1001(preprocess=True):
    sample = np.load('T1001.npz')
    sample_dist = sample['dist']
    sample_omega = sample['omega']
    sample_theta = sample['theta']
    sample_phi = sample['phi']
    seq = parse_a3m('T1001.a3m')[0]
    
    if not preprocess:
        return sample_dist, sample_omega, sample_theta, sample_phi, seq
    
    else:
        dist = logits2value(sample_dist, [None] + list(np.linspace(2,20,37)))
        omega = logits2value(sample_omega, [None] + list(np.linspace(-180,180, 24))) 
        theta = logits2value(sample_theta, [None] + list(np.linspace(-180,180, 24)))
        phi = logits2value(sample_theta, [None] + list(np.linspace(0,180, 24)))
        
        return dist, omega, theta, phi, seq
