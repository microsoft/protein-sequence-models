import argparse

import torch.cuda
from tqdm import tqdm
import numpy as np
import pandas as pd

from sequence_models.utils import parse_fasta
from sequence_models.pretrained import load_model_and_alphabet

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('in_fpath')
parser.add_argument('out_dir')
parser.add_argument('--repr_layers', nargs='*', default=[-1])
parser.add_argument('--include', nargs='*', default=['mean'])
parser.add_argument('--device', default=None)
parser.add_argument('--batchsize', default=1, type=int)
args = parser.parse_args()

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
seqs, names = parse_fasta(args.in_fpath, return_names=True)
ells = [len(s) for s in seqs]
seqs = [[s] for s in seqs]
n_total = len(seqs)
repr_layers = []
for r in args.repr_layers:
    if r == 'logits':
        logits = True
    else:
        repr_layers.append(int(r))
if 'logp' in args.include:
    logps = np.empty(len(seqs))
with tqdm(total=n_total) as pbar:
    for i in range(0, n_total, args.batchsize):
        start = i
        end = start + args.batchsize
        bs = seqs[start:end]
        bn = names[start:end]
        bl = ells[start:end]
        # tokenize
        x = collater(bs)[0].to(device)
        # pass through the model
        results = model(x, repr_layers=repr_layers, logits=logits)
        if 'representations' in results:
            for layer, rep in results['representations'].items():
                for r, ell, name in zip(rep, bl, bn):
                    r = r[:ell]
                    if 'mean' in args.include:
                        torch.save(r.mean(dim=0).detach().cpu(),
                                   args.out_dir + '_'.join([name, args.model, str(layer), 'mean']) + '.pt')
                    if 'per_tok' in args.include:
                        torch.save(r.detach().cpu(),
                                   args.out_dir + '_'.join([name, args.model, str(layer), 'per_tok']) + '.pt')
        if logits:
            rep = results['logits']
            for r, ell, name, src in zip(rep, bl, bn, x):
                if 'per_tok' in args.include:
                    r = r[:ell]
                    torch.save(r.detach().cpu(), args.out_dir + '_'.join([name, args.model, 'logits']) + '.pt')
                if 'logp' in args.include:
                    r = r.log_softmax(dim=-1)[:ell]
                    logps[i] = r[torch.arange(len(src)), src].mean().detach().cpu().numpy()
        pbar.update(len(bs))
    if 'logp' in args.include:
        df = pd.DataFrame()
        df['name'] = names
        df['sequence'] = [s[0] for s in seqs]
        df['logp'] = logps
        out_fpath = args.out_dir + args.model + '_logp.csv'
        print('Writing results to ' + out_fpath)
        df = df.to_csv(out_fpath, index=False)



