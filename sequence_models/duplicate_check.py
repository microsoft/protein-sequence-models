import subprocess
import argparse
import pandas as pd
from tqdm import tqdm

"""
TO RUN:
    To install mmseqs:
        conda install -c conda-forge -c bioconda mmseqs2

    To build db, use DB.fasta and run in commandline:
        mmseqs createdb examples/DB.fasta targetDB
        mmseqs createindex targetDB tmp

"""

parser = argparse.ArgumentParser(description='check sequence similarities')
parser.add_argument('--db_path', type=str, default='', help='database directory')
parser.add_argument('--q_path', type=str, default='', help='query fasta file')
parser.add_argument('--o_dir', type=str, default='', help='dir to save output files')
parser.add_argument('--coverage', type=float, default=0.8, help='min coverage between query and database')
parser.add_argument('--method', type=str, default='pident', help='method used: pident or score')
parser.add_argument('--cutoff', type=float, default=0.5, help='e value to determine cutoff for similarity')

args = parser.parse_args()


subprocess.run(['mmseqs', 'easy-search',
                args.q_path,
                args.db_path, 
                args.o_dir + 'report.m8', 
                'tmp',
                '-s', '1',
                '--format-output', 'query,target,raw,pident,nident,qlen,alnlen',
                '--cov-mode', '0',
                '-c', str(args.coverage)])

print('PARSING RESULTS')

# parse through results
query_sim = []
db_sim = []
score_sim = []

df = pd.read_csv(args.o_dir + 'report.m8', header=None, delimiter='\t')

if args.method == 'score':
    df['score'] = df[2]/df[6]
    for i in tqdm(range(len(df))):
        score = df.iloc[i]['score']
        if score >= args.cutoff:
            score_sim.append(score)
            query_sim.append(int(df.iloc[i][0]))
            db_sim.append(int(df.iloc[i][1]))

if args.method == 'pident':
    for i in tqdm(range(len(df))):
        score = df.iloc[i][3]
        if score >= args.cutoff:
            score_sim.append(score)
            query_sim.append(int(df.iloc[i][0]))
            db_sim.append(int(df.iloc[i][1]))
            
hits_df = pd.DataFrame({'query':query_sim, 'db':db_sim, args.method:score_sim})
hits_df.to_csv(args.o_dir + 'hits.txt', index=False)

duplicates = list(set(query_sim))
with open(args.o_dir + 'duplicates.txt', 'w') as f:
    for item in duplicates:
        f.write("%s\n" % item)
