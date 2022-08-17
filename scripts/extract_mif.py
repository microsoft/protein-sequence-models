import argparse

import torch.cuda
from tqdm import tqdm
import pandas as pd
import numpy as np

from sequence_models.utils import parse_fasta
from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.pdb_utils import parse_PDB, process_coords


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('csv_fpath')
parser.add_argument('pdb_dir')
parser.add_argument('out_dir')
parser.add_argument('result')
parser.add_argument('--include', nargs='*', default=['mean'])
parser.add_argument('--device', default=None)
args = parser.parse_args()

# Check inputs
if args.model not in ['mif', 'mifst']:
    raise ValueError("Valid models ars 'mif' and 'mifst'.")
if args.result == 'logits':
    for inc in args.include:
        if inc not in ['per_tok', 'logp']:
            raise ValueError("logits can be included as 'per_tok' or as 'logp'.")
elif args.result == 'repr':
    for inc in args.include:
        if inc not in ['per_tok', 'mean']:
            raise ValueError("repr can be included as 'per_tok' or as 'mean'.")
else:
    raise ValueError("Valid results ars 'repr' and 'logits'.")

# load model
print('Loading model...')
model, collater = load_model_and_alphabet(args.model)
# detect device and move model to it
if args.device is None:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
else:
    device = torch.device(args.device)
model = model.to(device)
# load data
print('Loading data...')
df = pd.read_csv(args.csv_fpath).reset_index()
if 'logp' in args.include:
    logps = np.empty(len(df))
with tqdm(total=len(df)) as pbar:
    for i, row in df.iterrows():
        seq = row['sequence']
        pdb = row['pdb']
        name = row['name']
        coords, wt, _ = parse_PDB(args.pdb_dir + pdb)
        coords = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
        dist, omega, theta, phi = process_coords(coords)
        batch = [[seq, torch.tensor(dist, dtype=torch.float),
                  torch.tensor(omega, dtype=torch.float),
                  torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
        src, nodes, edges, connections, edge_mask = collater(batch)
        src = src.to(device)
        nodes = nodes.to(device)
        edges = edges.to(device)
        connections = connections.to(device)
        edge_mask = edge_mask.to(device)
        rep = model(src, nodes, edges, connections, edge_mask, result=args.result)[0]
        if args.result == 'repr':
            if 'mean' in args.include:
                torch.save(rep.mean(dim=0).detach().cpu(),
                           args.out_dir + '_'.join([name, args.model, 'mean']) + '.pt')
            if 'per_tok' in args.include:
                torch.save(rep.detach().cpu(),
                           args.out_dir + '_'.join([name, args.model, 'per_tok']) + '.pt')
        else:
            if 'logp' in args.include:
                rep = rep.log_softmax(dim=-1)
                logps[i] = rep[torch.arange(len(src[0])), src].mean().detach().cpu().numpy()
            if 'per_tok' in args.include:
                torch.save(rep.detach().cpu(), args.out_dir + '_'.join([name, args.model, 'logits']) + '.pt')
        pbar.update(1)
    if 'logp' in args.include:
        df['logp'] = logps
        out_fpath = args.out_dir + args.model + '_logp.csv'
        print('Writing results to ' + out_fpath)
        df = df.to_csv(out_fpath, index=False)


