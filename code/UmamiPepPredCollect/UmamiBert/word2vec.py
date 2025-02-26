import gensim
from gensim.models import Word2Vec
import os
import pandas as pd

parent_dir = os.path.abspath(os.path.dirname(__file__))
def CBOW_word2vec(peptides, save_path=parent_dir+"/CBOW_word2vec.model"):

    tokenized_peptides = [list(peptide) for peptide in peptides]
    model = Word2Vec(sentences=tokenized_peptides, vector_size=50, window=2, sg=0, min_count=1, workers=4)
    model.save(save_path)

task_name = "ummpep2024"
df = pd.read_csv(parent_dir+f'/{task_name}_processed.csv', sep='\t')
CBOW_word2vec(df['Sequence'].tolist())



# model = Word2Vec.load(parent_dir+"/CBOW_word2vec.model")
# embedding_A = model.wv['A']

