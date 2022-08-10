Pytorch modules and utilities for modeling biological sequence data.

Here we will demonstrate the application of several tools we hope will help with modeling biological sequences.

## Installation

```
$ pip install sequence-models
$ pip install git+https://github.com/microsoft/protein-sequence-models.git  # bleeding edge, current repo main branch

```

## Loading pretrained models

Models require PyTorch. We tested on `v1.9.0`, `v1.11.0`,and `1.12`. If you installed into a clean conda environment, you may also need to install pandas, scipy, and wget. 

To load a model:

```
from sequence_models.pretrained import load_model_and_alphabet

model, collater = load_model_and_alphabet('carp_640M')
```

Available models are

- `carp_600k`
- `carp_38M`
- `carp_76M`
- `carp_640M`
- `mif`
- `mifst`
- `bigcarp_esm1bfinetune`
- `bigcarp_esm1bfrozen`
- `bigcarp_random`


## Convolutional autoencoding representations of proteins (CARP)

We make available pretrained CNN protein sequence masked language models of various sizes. All of these have a ByteNet encoder architecture and are pretrained on the March 2020 release of UniRef50 using the same masked language modeling task as in BERT and ESM-1b.

CARP is described in this [preprint](https://doi.org/10.1101/2022.05.19.492714).

You can also download the weights manually from [Zenodo](https://doi.org/10.5281/zenodo.6368483). 

To encode a batch of sequences: 

```
seqs = [['MDREQ'], ['MGTRRLLP']]
x = collater(seqs)[0]  # (n, max_len)
rep = model(x)  # (n, max_len, d_model)
```

CARP also supports computing representations from arbitrary layers and the final logits.

```
rep = model(x, repr_layers=[0, 2, 32], logits=True)
```

### Compute embeddings in bulk from FASTA

We provide a script that efficiently extracts embeddings in bulk from a FASTA file. A cuda device is optional and will be auto-detected. The following command extracts the final-layer embedding for a FASTA file from the `CARP_640M` model:

```
$ python scripts/extract.py carp_640M examples/some_proteins.fasta \
    examples/results/some_proteins_emb_carp_640M/ \
    --repr_layers 0 32 33 logits --include mean per_tok
```
Directory `examples/results/some_proteins_emb_carp_640M/` now contains one `.pt` file per extracted embedding; use `torch.load()` to load them. `scripts/extract.py` has flags that determine what .pt files are included:

`--repr-layers` (default: final only) selects which layers to include embeddings from. `0` is the input embedding. `logits` is the per-token logits.

`--include` specifies what embeddings to save. You can use the following:

- `per_tok` includes the full sequence, with an embedding per amino acid (seq_len x hidden_dim).
- `mean` includes the embeddings averaged over the full sequence, per layer (only valid for representations).
- `logp` computes the average log probability per sequence and stores it in a csv (only valid for logits).

`scripts/extract.py` also has `--batchsize` and `--device` flags. For example, to use GPU 2 on a multi-GPU machine, pass `--device cuda:2`. The default is to use a batchsize of 1 and `cpu` if cuda is not detected or `cuda:0` if cuda is detected. 

## Masked Inverse Folding (MIF) and Masked Inverse Folding with Sequence Transfer (MIF-ST)

We make available pretrained masked inverse folding models with and without sequence pretraining transfer from CARP-640M.

MIF and MIF-ST are described in this [preprint](https://doi.org/10.1101/2022.05.25.493516)

You can also download the weights manually from [Zenodo](https://zenodo.org/record/6573779#.YqjXT-zMI-Q). 

To encode a sequence with its structure: 

```
from sequence_models.pdb_utils import parse_PDB, process_coords
coords, wt, _ = parse_PDB('examples/gb1_a60fb_unrelaxed_rank_1_model_5.pdb')
coords = {
        'N': coords[:, 0],
        'CA': coords[:, 1],
        'C': coords[:, 2]
    }
dist, omega, theta, phi = process_coords(coords)
batch = [[wt, torch.tensor(dist, dtype=torch.float),
          torch.tensor(omega, dtype=torch.float),
          torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
src, nodes, edges, connections, edge_mask = collater(batch)
# can use result='repr' or result='logits'. Default is 'repr'.
rep = model(src, nodes, edges, connections, edge_mask)  
```

### Compute embeddings in bulk from csv

We provide a script that efficiently extracts embeddings in bulk from a csv file. A cuda device is optional and will be auto-detected. The following command extracts the final-layer embedding for a FASTA file from the `mifst` model:

```
$ python scripts/extract_mif.py mifst examples/gb1s.csv \
    examples/ \
    examples/results/some_proteins_mifst/ \
    repr --include mean per_tok
```
Directory `examples/results/some_proteins_mifst/` now contains one `.pt` file per extracted embedding; use `torch.load()` to load them. `scripts/extract_mif.py` has flags that determine what .pt files are included:

The syntax is:
```
$ python script/extract_mif.py <model> <csv_fpath> <pdb_dir> <out_dir> <result> --include <pooling options>
```

The input csv should have columns for `name`, `sequence`, and `pdb`. The script looks in `pdb_dir` for the filenames in the `pdb` column. 

The options for `result` are `repr` or `logits`.

`--include` specifies what embeddings to save. You can use the following:

- `per_tok` includes the full sequence, with an embedding per amino acid (seq_len x hidden_dim).
- `mean` includes the embeddings averaged over the full sequence, per layer (only valid for representations).
- `logp` computes the average log probability per sequence and stores it in a csv (only valid for logits).


`scripts/extract.py` also has a `--device` flags. For example, to use GPU 2 on a multi-GPU machine, pass `--device cuda:2`. The default is to use `cpu` if cuda is not detected or `cuda:0` if cuda is detected. 


## Biosynthetic gene cluster CARP (BiGCARP)

We make available pretrained CNN Pfam domain masked language models of BGCs. All of these have a ByteNet encoder architecture and are pretrained on antiSMASH using the same masked language modeling task as in BERT and ESM-1b.

BiGCARP is described in this [preprint](https://doi.org/10.1101/2022.07.22.500861). Training code is available [here](https://github.com/microsoft/protein-sequence-models).

You can also download the weights and datasets manually from [Zenodo](https://doi.org/10.5281/zenodo.6857704). 

To encode a batch of sequences: 

```
bgc = [['#;PF07690;PF06609;PF00083;PF00975;PF12697;PF00550;PF14765'],
       ['t3pks;PF07690;PF06609;PF00083;PF00975;PF12697;PF00550;PF14765;PF00698']]
model, collater = load_model_and_alphabet('bigcarp_esm1bfinetune')
x = collater(bgc)[0]
rep = model(x)
```


### Sequence Datasets and Dataloaders
In ```sampler.py```, you will find two Pytorch sampler classes: ```SortishSampler```, a sampler to sort similarly length 
 sample sequences into length-defined buckets; and ```ApproxBatchSampler```, a batch sampler which grabs sequences 
 from length-defined buckets until the batch has the set approximate max number of tokens or max number of tokens squared.
    
```
from sequence_models.samplers import SortishSampler, ApproxBatchSampler

# grab datasets
ds = dataset # your sequence dataset

# build dataloaders
len_ds = np.array([len(i[0]) for i in ds]) # list of lengths of the sequence in dataset (in order)
bucket_size = 1000 # number of length-defined buckets
max_tokens = 8000 # max number of tokens per batch
max_batch_size = 100 # max number of samples per batch
sortish_sampler = SortishSampler(len_ds, bucket_size)
batch_sampler = ApproxBatchSampler(sortish_sampler, max_tokens, max_batch_size, len_ds)
collater = collater # your collater function
dl = DataLoader(ds_train, collate_fn=collater, batch_sampler=batch_sampler, num_workers=16)
``` 

### Pre-implemented Models
* Struct2SeqDecoder (GNN)

The ```Struct2SeqDecoder``` model was adapted from 
[Ingraham et al.](https://papers.nips.cc/paper/2019/file/f3a4ff4839c56a5f460c88cce3666a2b-Paper.pdf). This model uses protein structural information 
encoded as a graph nodes and edges representing the structural information of each amino acid residue and their 
relations to each other, respectively.

If you already have node features, edge features, connections between nodes, encoded sequences (src), 
and edge mask (edge_mask); you can directly use the the ```Struct2SeqDecoder``` as demonstrated below:

```
from sequence_models.constants import trR_ALPHABET
from sequence_models.gnn import Struct2SeqDecoder

num_letters = len(trR_ALPHABET) # length of your amino acid alphabet  
node_features = 10 # number of node features
edge_features = 11 # number of edge features
hidden_dim =  128 # your choice of hidden layer dimension
num_decoder_layers = 3 # your choice of number of decoder layers to use
dropout = 0.1 # dropout used by decoder layer
use_mpnn = False # if True, use MPNN layer, else use Transformer layer for decoder 
direction = 'bidirectional' # direction of information flow/masking: forward, backward or bidirectional 

model = Struct2SeqDecoder(num_letters, node_features, edge_features, hidden_dim,
            num_decoder_layers, dropout, use_mpnn, direction)
out = model(nodes, edges, connections, src, edge_mask)
```

If you do not have prepared inputs, but have 2d maps representing the distance between residues (dist) and the dihedral
angles between residues (omega, theta, and phi), you can use our preprocessing functions to generate nodes, edges, and 
connections as demonstrated below:

```
from sequence_models.gnn import get_node_features, get_k_neighbors, get_edge_features, \
    get_mask, replace_nan

# process features
node = get_node_features(omega, theta, phi) # generate nodes
dist = dist.fill_diagonal_(np.nan) # if the diagonal of dist tensor is not already filled with nans, it should 
                                    # to prevent selecting self when getting k nearest residues in the next step 
connections = get_k_neighbors(dist, n_connections) # get connections
edge = get_edge_features(dist, omega, theta, phi, connections) # generate edge
edge_mask = get_mask(edge) # get edge mask (in the scenario where there is missing edge features between neighbors)
edge = replace_nan(edge) # replace nans with 0s 
node = replace_nan(node) 
```

Alternatively, we have also prepared ```StructureCollater```, a collater function 
found in ```collaters.py``` that also performs this task:

```
from sequence_models.collaters import StructureCollater

n_connections = 20 # number of connections per amino acid residue  
collater = StructureCollater(n_connections=n_connections)
ds = dataset # Dataset must return sequences, dists, omegas, thetas, phis 
dl = Dataloader(ds, collate_fn=collater)
```

* ByteNet

The ```ByteNet``` model was adapted from [Kalchbrenner et al.](https://arxiv.org/abs/1610.10099). ByteNet uses stacked
convolutional encoder and decoder layers to preserve temporal resolution of 
sequential data. 

```
from sequence_models.convolutional import ByteNet
from sequence_models.constants import trR_ALPHABET

n_tokens = len(trR_ALPHABET) # number of tokens in token dictionary
d_embedding = 128 # dimension of embedding
d_model = 128 # dimension to use within ByteNet model, //2 every layer
n_layers = 3 # number of layers of ByteNet block
kernel_size = 3 # the kernel width
r = ??? # used to calculate dilation factor
padding_idx = trR_ALPHABET.index('-') # location of padding token in ordered alphabet
causal = True # if True, chooses MaskedCausalConv1d() over MaskedConv1d()
dropout = 0.1 

x = torch.randn(32, 128) # input (n samples, len of seqs) 
input_mask = torch.ones(32, 128, 1) # mask (n samples, len of seqs, 1)
model = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, 
            padding_idx=padding_idx, causal=causal, dropout=dropout)
out = model(x, input_mask) 
```

We have also an implemented versions of ```ByteNet``` to be able to use 2d inputs (```ByteNet2d```) 
and as a language model (```ByteNetLM```): 

```
from sequence_models.convolutional import ByteNet2d, ByteNetLM

x = torch.randn(32, 128, 128, 64) # input (n samples, len of seqs, len of seqs, feature dimension)
input_mask = torch.ones # (n samples, len of seqs, len of seqs, 1), optional
model = ByteNet2d(d_in, d_model, n_layers, kernel_size, r, dropout=0.0)
out = model(x, input_mask)

x = torch.randn(32, 128) # input (n samples, len of seqs) 
input_mask = torch.ones(32, 128, 1) # mask (n samples, len of seqs, 1)
model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                    padding_idx=None, causal=False, dropout=0.0)
out = model(x, input_mask)
```

* trRosetta
The ```trRosetta``` model was implemented according to [Yang et al.](https://www.pnas.org/content/117/3/1496). In this model, multiple sequence
alignments (MSAs) are used to predict distances between amino acid residues as well as their dihedral 
angles (omega, theta, phi). Predictions are in the format of bins. Omega, theta and phi angle are binned into 24, 24, and 12 bins, respectively 
with 15 degrees segments and one no-contact bin. [Yang et al.](https://www.pnas.org/content/117/3/1496) has pretrained five models (model ids: 'a', 'b', 'c', 'd', 'e'). To run
a single model: 

```
from sequence_models.trRosetta_utils import trRosettaPreprocessing, parse_a3m
from sequence_models.trRosetta import trRosetta
from sequence_models.constants import trR_ALPHABET

msas = parse_a3m(path_to_msa) # load in msas in a3m format
alphabet = trR_ALPHABET # load your alphabet order
tr_preprocessing = trRosettaPreprocessing(alphabet) # setup preprocessor for msa
msas_processed = tr_preprocessing.process(msas)

n2d_layers = 61 # keep at 61 if you want to use pretrained version
model_id = 'a' # choose your pretrained model id
decoder = True # if True, return 2d structure maps, else returns hidden layer
p_dropout = 0.0
model = trRosetta(n2d_layers, model_id, decoder, p_dropout)
out = model(msas_processed) # returns dist_probs, theta_probs, phi_probs, omega_probs
```

To run an ensemble of models: 
```
from sequence_models.trRosetta_utils import trRosettaPreprocessing, parse_a3m
from sequence_models.trRosetta import trRosetta, trRosettaEnsemble
from sequence_models.constants import trR_ALPHABET

msas = parse_a3m(path_to_msa) # load in msas in a3m format
alphabet = trR_ALPHABET # load your alphabet order
tr_preprocessing = trRosettaPreprocessing(alphabet) # setup preprocessor for msa
msas_processed = tr_preprocessing.process(msas)

n2d_layers = 61 # keep at 61 if you want to use pretrained version
model_ids = 'abcde' # choose your pretrained model id
decoder = True # if True, return 2d structure maps, else returns hidden layer
p_dropout = 0.0
base_model = trRosetta
model = trRosettaEnsemble(base_model, n2d_layers, model_ids)
out = model(msas_processed)
```

If you would like to convert bin prediction into actual values, use ```probs2value```. 
Here is an example of converting distance bin predictions into values: 

```
from sequence_models.trRosetta_utils import probs2value

dist_probs, theta_probs, phi_probs, omega_probs  = model(x) # structure predictions (batch, # of bins, len of seq, len of seq)
preperty = 'dist' # choose between 'dist', 'theta', 'phi', or 'omega'
mask = mask # your 2d mask (batch, len of seq, len of seq)
dist_values = probs2value(dist, property, mask):

```

