
import os 
import torch
import esm
import pandas as pd
import collections
import numpy as np
import os
import shutil


parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


ESM_MODELS_INFO = {
    "esm2_t6_8M_UR50D": {"layers": 6, "hidden_size": 320},
    "esm2_t12_35M_UR50D": {"layers": 12, "hidden_size": 480},
    "esm2_t30_150M_UR50D": {"layers": 30, "hidden_size": 640},
    "esm2_t33_650M_UR50D": {"layers": 33, "hidden_size": 1280},
    "esm2_t36_3B_UR50D": {"layers": 36, "hidden_size": 2560},
    "esm2_t48_15B_UR50D": {"layers": 48, "hidden_size": 5120}, 
    "esm1v_t33_650M_UR90S_1": {"layers": 33, "hidden_size": 1280}, 
}



def esm_embeddings(tuple_list, esm_model_name='esm1v_t33_650M_UR90S_1',save_dir='single_mut/WT'):
    # https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t6_8M_UR50D-contact-regression.pt
    # https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t12_35M_UR50D-contact-regression.pt
    # https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t30_150M_UR50D-contact-regression.pt
    # https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt    
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(f'{parent_dir}/{esm_model_name}.pt')
    batch_converter = alphabet.get_batch_converter()
    model.eval()  
    batch_labels, batch_strs, batch_tokens = batch_converter(tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[ESM_MODELS_INFO[esm_model_name]["layers"]], return_contacts=True)  

    token_representations = results["representations"][ESM_MODELS_INFO[esm_model_name]["layers"]]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        # np.savetxt(f'{save_dir}/{batch_labels[i]}.txt', np.array(token_representations[i, 1 : tokens_len - 1]))
        # np.savez(f'{save_dir}/{batch_labels[i]}.npz', data=np.round(token_representations[i, 1:tokens_len-1], 4))
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    return embeddings_results



def generate_esm_embeddings(tuple_ls, esm_model_name, batch_size=50, temp_output_dir='output', save_dir='single_mut/WT'):
    os.makedirs(temp_output_dir, exist_ok=True)
    tasks = list(range(0, len(tuple_ls), batch_size))
    output_files = []
    for i in range(len(tasks)):
        print(f'Processing task {i+1}/{len(tasks)}')
        if i != (len(tasks) - 1):
            embeddings = np.array(esm_embeddings(tuple_ls[tasks[i]:tasks[i+1]], esm_model_name, save_dir))
        else:
            embeddings = np.array(esm_embeddings(tuple_ls[tasks[i]:], esm_model_name, save_dir))
        output_file = os.path.join(temp_output_dir, f'task_{i}.txt')
        np.savetxt(output_file, embeddings)
        output_files.append(output_file)
    features = np.vstack([np.loadtxt(f) for f in output_files])
    shutil.rmtree(temp_output_dir)
    return features
