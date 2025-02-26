import pandas as pd
import os

parent_dir = os.path.abspath(os.path.dirname(__file__))

df = pd.read_csv(parent_dir+'/ummpep2024_15.csv', sep='\t')
df = df[df['Length'] > 1]
df.to_csv(parent_dir+'/ummpep2024_processed_2-15.csv', sep='\t')
with open(parent_dir+'/ummpep2024_processed_2-15.fasta', 'w') as f:
    for index,row in df.iterrows():
        f.write('>'+row['Sequence']+'\n')
        f.write(row['Sequence']+'\n')
        

# git clone https://github.com/Superzchen/iFeature
# python iFeature/iFeature.py --file ummpep2024_processed_2-15.fasta --type PAAC