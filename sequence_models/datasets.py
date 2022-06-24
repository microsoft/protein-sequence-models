from typing import Union
from pathlib import Path
import lmdb
import subprocess
import string
import json
import os
from os import path
import pickle as pkl
from scipy.spatial.distance import squareform, pdist

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from sequence_models.utils import Tokenizer
from sequence_models.constants import trR_ALPHABET, DIST_BINS, PHI_BINS, THETA_BINS, OMEGA_BINS
from sequence_models.gnn import bins_to_vals
from sequence_models.pdb_utils import process_coords

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class TAPEDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 data_type: str,
                 split: str,
                 sub_type : str = 'distance',
                 eps : float = 1e-6,
                 in_memory: bool = False,
                 max_len=700):

        """
        data_path : path to data directory

        data_type : name of downstream task, [fluorescence, stability, remote_homology, 
            secondary_structure, contact]
        
        split : data split to load

        contact_method : if data_type == contact, choose 'distance' to get 
            distance instead of binary contact output
        """
        
        self.data_type = data_type
        self.sub_type = sub_type
        self.eps = eps
        self.max_len = max_len
        
        if data_type == 'fluorescence':
            if split not in ('train', 'valid', 'test'):
                raise ValueError(f"Unrecognized split: {split}. "
                                 f"Must be one of ['train', 'valid', 'test']")

            data_file = Path(data_path + f'fluorescence_{split}.lmdb')
            self.output_label = 'log_fluorescence'
            
        if data_type == 'stability':
            if split not in ('train', 'valid', 'test'):
                raise ValueError(f"Unrecognized split: {split}. "
                                 f"Must be one of ['train', 'valid', 'test']")

            data_file = Path(data_path + f'stability_{split}.lmdb')
            self.output_label = 'stability_score'
        
        if data_type == 'remote_homology':
            if split not in ('train', 'valid', 'test_fold_holdout',
                             'test_family_holdout', 'test_superfamily_holdout'):
                raise ValueError(f"Unrecognized split: {split}. Must be one of "
                                 f"['train', 'valid', 'test_fold_holdout', "
                                 f"'test_family_holdout', 'test_superfamily_holdout']")

            data_file = Path(data_path + f'remote_homology_{split}.lmdb')
            self.output_label = 'fold_label'
            
        if data_type == 'secondary_structure':
            if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
                raise ValueError(f"Unrecognized split: {split}. Must be one of "
                                 f"['train', 'valid', 'casp12', "
                                 f"'ts115', 'cb513']")

            data_file = Path(data_path + f'secondary_structure_{split}.lmdb')
            if self.sub_type == 'ss8':
                self.output_label = 'ss8'
            else:
                self.output_label = 'ss3'
            
        if data_type == 'contact':
            if split not in ('train', 'train_unfiltered', 'valid', 'test'):
                raise ValueError(f"Unrecognized split: {split}. Must be one of "
                                 f"['train', 'train_unfiltered', 'valid', 'test']")

            data_file = Path(data_path + f'proteinnet_{split}.lmdb')
            self.output_label = 'tertiary'
            
        self.data = LMDBDataset(data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        primary = item['primary']
        mask = None
        
        if self.data_type in ['fluorescence', 'stability', ]:
            output = float(item[self.output_label][0])
        
        if self.data_type in ['remote_homology']:
            output = item[self.output_label]
            diff = max(len(primary) - self.max_len + 1, 1)
            start = np.random.choice(diff)
            end = start + self.max_len
            primary = primary[start: end]

        if self.data_type in ['secondary_structure']:
            # pad with -1s because of cls/sep tokens
            output = torch.Tensor(item[self.output_label],).to(torch.int8)
            diff = max(len(primary) - self.max_len + 1, 1)
            start = np.random.choice(diff)
            end = start + self.max_len
            primary = primary[start: end]
            output = output[start:end]

        if self.data_type in ['contact']:
            # -1 is ignore, 0 in no contact, 1 is contact
            valid_mask = item['valid_mask']
            distances = squareform(pdist(item[self.output_label]))
            yind, xind = np.indices(distances.shape)
            invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
            invalid_mask |= np.abs(yind - xind) < 6
            if self.sub_type == 'distance':
                output = torch.tensor(np.exp(-distances ** 2 / 64))
            else:
                contact_map = np.less(distances, 8.0).astype(np.int64)
                contact_map[invalid_mask] = -1
                contact_map = torch.Tensor(contact_map).to(torch.int8)
                output = torch.tensor(contact_map)
            mask = torch.tensor(~invalid_mask)
            diff = max(len(primary) - self.max_len + 1, 1)
            start = np.random.choice(diff)
            end = start + self.max_len
            primary = primary[start: end]
            output = output[start:end, start:end]
            mask = mask[start:end, start:end]
        return primary, output, mask


class CSVDataset(Dataset):

    def __init__(self, fpath=None, df=None, split=None, outputs=[], max_len=np.inf):
        if df is None:
            self.data = pd.read_csv(fpath)
        else:
            self.data = df
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        self.outputs = outputs
        self.data = self.data[['sequence'] + self.outputs]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        if len(sequence) > self.max_len:
            start = np.random.choice(len(sequence) - self.max_len)
            stop = start + self.max_len
            sequence = sequence[start:stop]
        return [sequence, *row[self.outputs]]


class FlatDataset(Dataset):

    def __init__(self, fpath, offsets, cols=[1]):
        self.fpath = fpath
        self.offsets = offsets
        self.cols = cols

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.fpath, 'r') as f:
            f.seek(self.offsets[idx])
            line = f.readline()[:-1]  # strip the \n
            line = line.split(',')
            return [line[i] for i in self.cols]


class FFDataset(Dataset):

    def __init__(self, stem, max_len=np.inf, tr_only=True):
        self.index = stem + 'ffindex'
        self.data = stem + 'ffdata'
        result = subprocess.run(['wc', '-l', self.index], stdout=subprocess.PIPE)
        self.length = int(result.stdout.decode('utf-8').split(' ')[0])
        self.tokenizer = Tokenizer(trR_ALPHABET)
        self.table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
        self.max_len = max_len
        self.tr_only = tr_only

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        result = subprocess.run(['ffindex_get', self.data, self.index, '-n', str(idx + 1)],
                                stdout=subprocess.PIPE)
        a3m = result.stdout.decode('utf-8')
        seqs = []
        for line in a3m.split('\n'):
            # skip labels
            if len(line) == 0:
                continue
            if line[0] == '#':
                continue
            if line[0] != '>':
                # remove lowercase letters and right whitespaces
                s = line.rstrip().translate(self.table)
                if self.tr_only:
                    s = ''.join([a if a in trR_ALPHABET else '-' for a in s])
                if len(s) > self.max_len:
                    return torch.tensor([])
                seqs.append(s)
        seqs = torch.tensor([self.tokenizer.tokenize(s) for s in seqs])
        return seqs


def trr_bin(dist, omega, theta, phi):
    dist = torch.tensor(np.digitize(dist, DIST_BINS[1:]) % (len(DIST_BINS) - 1))
    idx = np.where(omega == omega)
    jdx = np.where(omega[idx] < 0)[0]
    idx = tuple(i[jdx] for i in idx)
    omega[idx] = 2 * np.pi + omega[idx]
    omega = torch.tensor(np.digitize(omega, OMEGA_BINS[1:]) % (len(OMEGA_BINS) - 1))
    idx = np.where(theta == theta)
    jdx = np.where(theta[idx] < 0)[0]
    idx = tuple(i[jdx] for i in idx)
    theta[idx] = 2 * np.pi + theta[idx]
    theta = torch.tensor(np.digitize(theta, THETA_BINS[1:]) % (len(THETA_BINS) - 1))
    phi = torch.tensor(np.digitize(phi, PHI_BINS[1:]) % (len(PHI_BINS) - 1))
    idx = torch.where(dist == 0)
    omega[idx] = 0
    theta[idx] = 0
    phi[idx] = 0
    return dist, omega, theta, phi


class UniRefDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(self, data_dir: str, split: str, structure=False, pdb=False, coords=False, bins=False,
                 p_drop=0.0, max_len=2048):
        self.data_dir = data_dir
        self.split = split
        self.structure = structure
        self.coords = coords
        with open(data_dir + 'splits.json', 'r') as f:
            self.indices = json.load(f)[self.split]
        metadata = np.load(self.data_dir + 'lengths_and_offsets.npz')
        self.offsets = metadata['seq_offsets']
        self.pdb = pdb
        self.bins = bins
        if self.pdb or self.bins:
            self.n_digits = 6
        else:
            self.n_digits = 8
        if self.coords:
            with open(data_dir + 'coords.pkl', 'rb') as f:
                self.structures = pkl.load(f)
        self.p_drop = p_drop
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        with open(self.data_dir + 'consensus.fasta') as f:
            f.seek(offset)
            consensus = f.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        if self.coords:
            coords = self.structures[str(idx)]
            dist, omega, theta, phi = process_coords(coords)
            dist = torch.tensor(dist).float()
            omega = torch.tensor(omega).float()
            theta = torch.tensor(theta).float()
            phi = torch.tensor(phi).float()
        elif self.structure:
            sname = 'structures/{num:{fill}{width}}.npz'.format(num=idx, fill='0', width=self.n_digits)
            fname = self.data_dir + sname
            if path.isfile(fname):
                structure = np.load(fname)
            else:
                structure = None
            if structure is not None:
                if np.random.random() < self.p_drop:
                    structure = None
                elif self.pdb:
                    dist = torch.tensor(structure['dist']).float()
                    omega = torch.tensor(structure['omega']).float()
                    theta = torch.tensor(structure['theta']).float()
                    phi = torch.tensor(structure['phi']).float()
                    if self.bins:
                        dist, omega, theta, phi = trr_bin(dist, omega, theta, phi)
                else:
                    dist, omega, theta, phi = bins_to_vals(data=structure)
            if structure is None:
                dist, omega, theta, phi = bins_to_vals(L=len(consensus))
        if self.structure or self.coords:
            consensus = consensus[start:stop]
            dist = dist[start:stop, start:stop]
            omega = omega[start:stop, start:stop]
            theta = theta[start:stop, start:stop]
            phi = phi[start:stop, start:stop]
            return consensus, dist, omega, theta, phi
        consensus = consensus[start:stop]
        return (consensus, )


class TRRDataset(Dataset):
    def __init__(self, data_dir, dataset, return_msa=True, bin=True, untokenize=False, max_len=2048):
        """
        Args:
            data_dir: str,
                path to trRosetta data
            dataset: str,
                train, valid
            return_msa: bool
                return full MSA or single sequence
            bin: bool
                bin structure matrices
            tokenizer:
                Use this to untokenize sequence if desired
        """
        filenames = data_dir + dataset + 'list.txt'
        self.filenames = np.loadtxt(filenames, dtype=str)
        self.data_dir = data_dir
        self.return_msa = return_msa
        self.bin = bin
        self.max_len = max_len
        if untokenize:
            self.tokenizer = Tokenizer(trR_ALPHABET)
        else:
            self.tokenizer = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.data_dir + 'npz/' + self.filenames[idx] + '.npz'
        data = np.load(filename)
        if self.return_msa:
            s = torch.tensor(data['msa'])
            ell = s.shape[1]
        else:
            s = data['msa'][0]
            if self.tokenizer is not None:
                s = self.tokenizer.untokenize(s)
            ell = len(s)
        if ell - self.max_len > 0:
            start = np.random.choice(ell - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = ell
        dist = data['dist6d']
        omega = data['omega6d']
        theta = data['theta6d']
        phi = data['phi6d']
        if self.return_msa:
            s = s[:, start:stop]
        else:
            s = s[start:stop]
        if self.bin:
            dist, omega, theta, phi = trr_bin(dist, omega, theta, phi)
        else:
            idx = np.where(dist == 0)
            dist[idx] = 20.0
            dist = torch.tensor(dist).float()
            omega = torch.tensor(omega).float()
            theta = torch.tensor(theta).float()
            phi = torch.tensor(phi).float()
        dist = dist[start:stop, start:stop]
        omega = omega[start:stop, start:stop]
        theta = theta[start:stop, start:stop]
        phi = phi[start:stop, start:stop]

        return s, dist, omega, theta, phi

    
    
class MSAGapDataset(Dataset):
    """Build dataset for trRosetta data: gap-prob and lm/mlm"""
    def __init__(self, data_dir, dataset, task, pdb=False, y=None, msa=None, 
                 random_seq=False, npz_dir=None, reweight=True, mask_endgaps=False):
        """
        Args:
            data_dir: str,
                path to trRosetta data
            dataset: str,
                train, valid, or test
            task: str,
                gap-prob or lm
            pdb: bool,
                if True, return structure as inputs; if False, return random sequence
                if pdb is False, you must have task = gab-prob
            filtered_y: bool,
                if True, use gap probabilities from filtered MSA (task = gap-prob)
                or select sequence from filtered MSA (task = lm)
                if False, use unfiltered MSA
                Filtered is defined as removing sequences from MSA where gaps only
                exists on ends of sequences
            filtered_msa: bool,
                if True, use filtered msa; if False, use unfiltered msa
            npz_dir: str,
                if you have a specified npz directory
            pdb_dir: str,
                if you have a specified pdb directory
        """
        filename = data_dir + dataset + 'list.txt'
        pdb_ids = np.loadtxt(filename, dtype=str)

        # choose to use specific msa or y instead of the prebuilt ones
        # should be in the order of npz_dir and in npy file
        self.msa_path = msa
        self.y_path = y

        # get data dir
        if npz_dir:
            self.npz_dir = npz_dir
        else:
            self.npz_dir = data_dir + 'structure/'
        all_npzs = os.listdir(self.npz_dir)
        selected_npzs = [i for i in pdb_ids if i + '.npz' in all_npzs]
        self.filenames = selected_npzs  # ids of samples to include
        
        # X options
        self.pdb = pdb
        self.task = task
        self.random_seq = random_seq
        
        # special options for generating y values
        self.reweight = reweight
        self.mask_endgaps = mask_endgaps 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(self.npz_dir + filename + '.npz')
        
        # grab sequence info
        if self.msa_path is not None:
            msa_data = np.load(self.msa_path + filename + ".npz") 
            msa = msa_data['msa']
            weights = msa_data['weights']
        else:
            msa = data['msa']
            weights = data['weights']
        anchor_seq = msa[0]
        if self.random_seq:
            flag = True
            while flag:
                random_idx = np.random.randint(0, len(msa))
                base_seq = msa[random_idx]
                if (base_seq == 20).sum()/len(base_seq) < 0.20:
                    flag = False
        else:
            base_seq = anchor_seq

        # choose y type
        if self.y_path is not None:
            y_data = np.load(self.y_path + filename + 'npz')
            y = y_data['y']
            y_mask = y_data['y_mask']
        elif self.task == "gap-prob":
            if self.reweight: # downsampling
                y = ((msa == 20) * weights.T).sum(0)/msa.shape[0]
                y = torch.FloatTensor(y)
            else:
                y = torch.FloatTensor(np.sum(msa == 20, axis=0) / msa.shape[0])
            y_mask = None
        else:  # lm
#             y, y_mask = self._get_lm_y(msa)
            y = torch.LongTensor(base_seq)
            y_mask = None
        # choose X type
        if self.pdb:  # use structure for X
            dist = torch.FloatTensor(data['dist'])
            omega = torch.FloatTensor(data['omega'])
            theta = torch.FloatTensor(data['theta'])
            phi = torch.FloatTensor(data['phi'])
            base_seq = torch.LongTensor(base_seq)
            anchor_seq = torch.LongTensor(anchor_seq)
            return base_seq, anchor_seq, dist, omega, theta, phi, y, y_mask

        else:  # use just seq for X (THIS IS ONLY USED FOR GAP PROB)
            if self.task == "gap-prob":
                chosen = False
                while not chosen:
                    msa_num = np.random.randint(msa.shape[0])
                    x = msa[msa_num]
                    seq_mask = x != 20
                    # only want num of gap < 20%
                    chosen = np.sum(seq_mask) / x.shape[0] > 0.8
                x = torch.LongTensor(x)
                seq_mask = torch.BoolTensor(seq_mask)
                return x[seq_mask], y[seq_mask]
            else:
                raise ValueError("""Warning - input type and output type are not compatible, 
                    pdb=False can only be used with task gap-prob""")

    def _get_lm_y(self, msa):
        if self.mask_endgaps:
            y = torch.LongTensor(msa[np.random.choice(msa.shape[0])])  # get random seq from msa
            y_mask = []
            for i in range(len(y)):
                if y[i] != 20:
                    y_mask += [False] * (i)
                    break
            for j in range(len(y) - 1, -1, -1):
                if y[j] != 20:
                    y_mask += [True] * (j - i + 1)
                    y_mask += [False] * (len(y) - 1 - j)
                    break
            return y, torch.BoolTensor(y_mask)
        else:
            y = torch.LongTensor(msa[np.random.choice(msa.shape[0])])  # get random seq from msa
            y_mask = None
            return y, y_mask


class TRRMSADataset(Dataset):
    """Build dataset for trRosetta data: MSA Absorbing Diffusion model"""

    def __init__(self, n_sequences=64, npz_dir=None):
        """
        Args:
            n_sequences: int,
                number of sequences to subsample down to
            npz_dir: str,
                if you have a specified npz directory
        """

        # Get npz_data dir
        if npz_dir is not None:
            self.npz_dir = npz_dir
        else:
            raise FileNotFoundError(npz_dir)

        # MSAs should be in the order of npz_dir
        all_files = os.listdir(self.npz_dir)
        self.filenames = all_files  # IDs of samples to include

        # Number of sequences to subsample down to
        self.n_sequences = n_sequences

        self.tokenizer = Tokenizer(trR_ALPHABET)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # TODO: add error checking?
        filename = self.filenames[idx]
        data = np.load(self.npz_dir + filename)

        # Grab sequence info
        msa = data['msa']
        msa = [self.tokenizer.untokenize(s) for s in msa]  # Untokenize sequences, msa is a list of strings

        anchor_seq = msa[0]  # This is the query sequence in MSA

        # TODO: keep "unique" sequences when subsampling rather than random
        MSA_num_seqs = len(msa)
        output = []

        # If fewer sequences in MSA than self.n_sequences,
        # create sequences padded with STOP token
        if MSA_num_seqs < self.n_sequences:
            diff = self.n_sequences - MSA_num_seqs
            MSA_seq_len = len(anchor_seq)
            padded_seq = [STOP] * MSA_seq_len
            output.append(msa)
            for i in range(diff):
                output.append(padded_seq)
        elif MSA_num_seqs == self.n_sequences:
            output.append(msa)
        else:
            random_idx = np.random.choice(MSA_num_seqs - 1, size=self.n_sequences - 1, replace=False)
            random_idx += 1
            sample_seq = np.array(msa)[random_idx]
            output.append(anchor_seq)
            # output.append(sample_seq)
            for seq in sample_seq:
                output.append(seq)

        return list(output)


class A3MMSADataset(Dataset):
    """Build dataset for A3M data: MSA Absorbing Diffusion model"""

    def __init__(self, split, n_sequences=64, data_dir=None):
        """
        Args:
            n_sequences: int,
                number of sequences to subsample down to
            data_dir: str,
                if you have a specified data directory
        """

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        self.split = split

        if self.split == 'valid':
            self.data_dir += 'valid/'
        else:
            self.data_dir += 'test/'

        all_files = os.listdir(self.data_dir)
        self.filenames = all_files  # IDs of samples to include

        # Number of sequences to subsample down to
        self.n_sequences = n_sequences

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # TODO: add error checking?
        filename = self.filenames[idx]
        parsed_msa = parse_fasta(self.data_dir + filename, idx)

        anchor_seq = parsed_msa[0]  # This is the query sequence in MSA

        # TODO: keep "unique" sequences when subsampling rather than random
        MSA_num_seqs = len(parsed_msa)
        output = []

        # If fewer sequences in MSA than self.n_sequences,
        # create sequences padded with STOP token
        if MSA_num_seqs < self.n_sequences:
            diff = self.n_sequences - MSA_num_seqs
            MSA_seq_len = len(anchor_seq)
            padded_seq = [STOP] * MSA_seq_len
            output += parsed_msa
            for i in range(diff):
                output.append(padded_seq)
        elif MSA_num_seqs == self.n_sequences:
            output += parsed_msa
        else:
            random_idx = np.random.choice(MSA_num_seqs - 1, size=self.n_sequences - 1, replace=False)
            random_idx += 1
            sample_seq = np.array(parsed_msa)[random_idx]
            output.append(anchor_seq)
            # output.append(sample_seq)
            for seq in sample_seq:
                output.append(seq)

        return list(output)
