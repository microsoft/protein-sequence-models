import json
import os
import numpy as np
import pandas as pd
import wget
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class AAIndexTokenizer(object):
    """Convert between strings and their AAIndex representations."""
    def __init__(self, dpath: str, n_comp: int = 20):
        """
        Args:
            dpath: directory to save raw and reduced representations
            n_comp: number of components in PCA
        """
        alphabet = AAINDEX_ALPHABET
        if not os.path.exists(dpath):
            os.mkdir(dpath)
        if not os.path.exists(dpath + '/aaindex1'):
            file = wget.download('ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1',
                                 out=dpath + '/' + 'aaindex1')
        if not os.path.exists(dpath + '/raw_aaindex.json'):
            raw_dict = {i: [] for i in alphabet}
            with open(dpath + '/aaindex1', 'r') as f:
                for line in f:
                    if line[0] == 'I':
                        set1 = next(f).strip().split()
                        set2 = next(f).strip().split()
                        set = set1 + set2
                        for i in range(len(alphabet)):
                            val = set[i]
                            if val == 'NA':
                                val = None
                            else:
                                val = float(val)
                            raw_dict[alphabet[i]].append(val)
            with open(dpath + '/raw_aaindex.json', 'w') as f:
                json.dump(raw_dict, f)
        if not os.path.exists(dpath + '/red_aaindex.json'):
            with open(dpath + '/raw_aaindex.json') as f:
                raw_dict = json.load(f)
            # preprocessing : drop embeddings with missing data (drop 13)
            embed_df = pd.DataFrame(raw_dict).dropna(axis=0)
            embed = embed_df.values.T  # (len(alphabet), 553)
            # scale to 0 mean and unit variance
            scaler = StandardScaler()
            embed = scaler.fit_transform(embed)
            # PCA
            pca = PCA(n_components=n_comp, svd_solver='auto')
            embed_red = pca.fit_transform(embed)
            print('VARIANCE EXPLAINED: ', pca.explained_variance_ratio_.sum())
            red_dict = {alphabet[i]: list(embed_red[i, :]) for i in range(len(alphabet))}
            with open(dpath + '/red_aaindex.json', 'w') as f:
                json.dump(red_dict, f)
        # save reduced representation
        with open(dpath + '/red_aaindex.json') as f:
            self.red_dict = json.load(f)
    def tokenize(self, seq: str) -> np.ndarray:
        """
        Args:
            seq: str
                amino acid sequence
        Returns:
            encoded: np.array
                encoded amino acid sequence based on reduced AAIndex representation, (L*n_comp,)
        """
        encoded = np.concatenate([self.red_dict[a] for a in seq])
        return encoded