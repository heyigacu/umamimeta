import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch, Selection
import os
import torch
import numpy as np
from Bio import PDB
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv
import torch.optim as optim
from sklearn.model_selection import train_test_split

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rec_dataset_dir = "dataset/umami_rec_dataset"
rec_taskname = 'ummrec'
lig_dataset_dir = "dataset/umami_pep_dataset"
lig_taskname = 'ummpep2024_9_15'
esm_model = "esm2_t6_8M_UR50D"
ESM_name = "esm2_t6_8M_UR50D"
result_dir = parent_parent_dir+f'/result/HG_{ESM_name}_{lig_taskname}/'
if not os.path.exists(result_dir): os.mkdir(result_dir)
task_dir = f"{lig_dataset_dir}/{lig_taskname}"

df_all_lig = pd.read_csv(f"{lig_dataset_dir}/ummpep2024/ummpep2024_processed.csv", sep="\t")

def load_structure(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    dic = {}
    for model in structure:
        for chain in model:
            for i,residue in enumerate(chain):
                if residue.has_id('CA'):
                    dic[i] = residue['CA'].get_coord()
    return dic

def distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def slice_dict_by_keys(d, keys):
    return {k: d[k] for k in keys if k in d}

def find_nearby_residues(protein_path, ligand_path, pro_embed_path, lig_embed_path, cutoff=10.0):
    protein_atoms = load_structure(protein_path)
    ligand_atoms = load_structure(ligand_path)
    pro_embed = np.load(pro_embed_path)['data']
    lig_embed = np.load(lig_embed_path)['data']

    ligand_protein_edges = []
    interacted_protein_atoms = []
    protein_protein_edges = []
    ligand_ligand_edges = []

    for i,lig_atom_coord in ligand_atoms.items():
        for j,pro_atom_coord in protein_atoms.items():
            dist = distance(lig_atom_coord, pro_atom_coord)
            if dist <= cutoff:
                ligand_protein_edges.append((i,j))
                interacted_protein_atoms.append(j)
        
        for k,lig_atom_coord_ in ligand_atoms.items():
            dist = distance(lig_atom_coord, lig_atom_coord_)
            if dist <= cutoff:
                ligand_ligand_edges.append((i, k))


    interacted_protein_atoms = list(set(interacted_protein_atoms))
    interacted_protein_atoms = slice_dict_by_keys(protein_atoms, interacted_protein_atoms)
    for i, pro_atom_coord in interacted_protein_atoms.items():
        for j, pro_atom_coord_ in interacted_protein_atoms.items():
            dist = distance(pro_atom_coord, pro_atom_coord_)
            if dist <= cutoff:
                protein_protein_edges.append((i,j))
    
    ligand_set = set()
    for edge in ligand_ligand_edges:
        ligand_set.update(edge)
    ligand_ls = list(ligand_set)
    dic_lig_ = dict(enumerate(ligand_ls))   
    lig_embed = lig_embed[ligand_ls]     
    dic_lig = {v: k for k, v in dic_lig_.items()}

    protein_set = set()
    for edge in protein_protein_edges:
        protein_set.update(edge)
    pro_ls = list(protein_set)
    dic_pro_ = dict(enumerate(pro_ls))
    pro_embed = pro_embed[pro_ls]    
    dic_pro = {v: k for k, v in dic_pro_.items()}

    reindex_ligand_protein_edges = []
    reindex_protein_protein_edges = []
    reindex_ligand_ligand_edges = []


    for lig,pro in ligand_protein_edges:
        reindex_ligand_protein_edges.append((dic_lig[lig], dic_pro[pro]))

    for pro1,pro2 in protein_protein_edges:
        reindex_protein_protein_edges.append((dic_pro[pro1], dic_pro[pro2]))
    
    for lig1,lig2 in ligand_ligand_edges:
        reindex_ligand_ligand_edges.append((dic_lig[lig1], dic_lig[lig2]))

    g = dgl.heterograph({
        ('ligand', 'interacts', 'protein'): reindex_ligand_protein_edges,
        ('protein', 'interacts', 'ligand'): [(j, i) for i, j in reindex_ligand_protein_edges],
        ('protein', 'interacts', 'protein'): reindex_protein_protein_edges,
        ('ligand', 'interacts', 'ligand'): reindex_ligand_ligand_edges
    })

    g.nodes['ligand'].data['feat'] = torch.from_numpy(lig_embed)
    g.nodes['protein'].data['feat'] = torch.from_numpy(pro_embed)
    return g


class HeteroGraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(HeteroGraphClassifier, self).__init__()
        self.gnn_layers = nn.ModuleList([
            HeteroGraphConv({
                ('ligand', 'interacts', 'protein'): GraphConv(in_feats, hidden_size),
                ('protein', 'interacts', 'ligand'): GraphConv(in_feats, hidden_size),
                ('protein', 'interacts', 'protein'): GraphConv(in_feats, hidden_size),
                ('ligand', 'interacts', 'ligand'): GraphConv(in_feats, hidden_size),
            })
            for _ in range(8) 
        ])
        self.fc = nn.Linear(hidden_size * 8, out_feats) 

    def forward(self, batched_graphs):
        node_feats = []
        for i in range(8):
            graphs = dgl.unbatch(batched_graphs[i])
            h_list = []
            for g in graphs:
                inputs = {
                    'ligand': g.nodes['ligand'].data['feat'],
                    'protein': g.nodes['protein'].data['feat']
                }
                h_dict = self.gnn_layers[i](g, inputs)
                h = h_dict['ligand']
                h_list.append(h.mean(0).unsqueeze(0))
            h_batch = torch.cat(h_list, dim=0) # mean readout: (batch,100)
            node_feats.append(h_batch)
        combined_feats = torch.cat(node_feats, dim=1) # mean readout: (batch,100)
        out = self.fc(combined_feats)
        return torch.sigmoid(out)

def load_graphs(df_rec):
    total_graphs = []
    for i_rec,row_rec in df_rec.iterrows():
        rec = row_rec['pocket']
        graphs, _ = dgl.load_graphs(f'dataset/hgs/{rec}.dgl')
        total_graphs.append(graphs)
    return list(map(list, zip(*total_graphs)))

def create_batch(samples):
    batched_graphs = []
    for i in range(8): 
        graphs = [sample[i] for sample in samples] 
        batched_graphs.append(dgl.batch(graphs)) 
    return batched_graphs

def generate_graphs():
    lig_embeddings_dir = f"{lig_dataset_dir}/{lig_taskname}/aa_{esm_model}"
    rec_embeddings_dir = f"{rec_dataset_dir}/{rec_taskname}/aa_{esm_model}"
    for i_rec,row_rec in df_rec.iterrows():
        rec = row_rec['pocket']
        pro_embed_path = rec_embeddings_dir+f'/{i_rec}.npz'
        graphs = []
        for i_lig,row_lig in df_lig.iterrows():
            print(i_lig)
            lig = row_lig['Sequence']
            if len(lig) < 5:
                lig_pdb = parent_parent_dir+f"/dock/dock/{rec}/{lig}/{rec}_{lig}.pdb"
            else:
                lig_pdb = parent_parent_dir+f"/dock/dock/{rec}/{lig}/{rec}_{lig}_ranked_1.pdb"
            rec_pdb = parent_parent_dir+f"/dock/receptor/pdb/{rec}.pdb"
            lig_embed_path = lig_embeddings_dir+f'/{i_lig}.npz'
            graph = find_nearby_residues(rec_pdb, lig_pdb, pro_embed_path, lig_embed_path)
            graphs.append(graph)
        dgl.save_graphs(f'dataset/hgs/{rec}.dgl', graphs)

def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()



device = torch.device('cpu')
print(f"Using device: {device}")
df_lig = pd.read_csv(f"{lig_dataset_dir}/{lig_taskname}/{lig_taskname}_processed.csv", sep="\t")
df_rec = pd.read_csv(f"{rec_dataset_dir}/{rec_taskname}/{rec_taskname}_processed.csv", sep="\t")
labels = list(df_all_lig['Label'])
graphs = load_graphs(df_rec)

for fold in range(1,6):
    train_ids =  load_ids(task_dir+f'/fold{fold}_train_ids.txt')
    train_seqs = [list(df_lig['Sequence'])[i] for i in train_ids]
    filtered_df = df_all_lig[df_all_lig['Sequence'].isin(train_seqs)]
    train_ids = filtered_df["ID"].to_list()

    test_ids =  load_ids(task_dir+f'/fold{fold}_test_ids.txt')
    test_seqs = [list(df_lig['Sequence'])[i] for i in test_ids]
    filtered_df = df_all_lig[df_all_lig['Sequence'].isin(test_seqs)]
    test_ids = filtered_df["ID"].to_list()

    train_graphs = [graphs[i] for i in train_ids]
    train_labels = [labels[i] for i in train_ids]
    test_graphs = [graphs[i] for i in test_ids]
    test_labels = [labels[i] for i in test_ids]
    model = HeteroGraphClassifier(in_feats=320, hidden_size=100, out_feats=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    num_epochs = 100
    batch_size = 64
    best_val_loss = float('inf')
    counter = 0
    patience = 7

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i in range(0, len(train_graphs), batch_size):
            batch_samples = train_graphs[i:i + batch_size]
            batched_graphs = create_batch(batch_samples)
            batch_labels = torch.tensor(train_labels[i:i + batch_size]).to(device)

            outputs = model(batched_graphs)
            loss = criterion(outputs.squeeze(-1), batch_labels.float())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / (i+1)

        test_loss = 0.0
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for i in range(0, len(test_graphs), batch_size):
                batch_samples = test_graphs[i:i + batch_size]
                batched_graphs = create_batch(batch_samples)
                batch_labels = torch.tensor(test_labels[i:i + batch_size]).to(device)
                outputs = model(batched_graphs)
                test_loss += criterion(outputs.squeeze(-1), batch_labels.float()).item()
                val_preds.append(outputs.squeeze(-1))
                val_labels.append(batch_labels)
        test_loss = test_loss / (i+1)
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        print(f'epoch:{epoch}, train_loss:{train_loss}, test_loss:{test_loss}')
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            counter = 0
            np.savetxt(result_dir+f'/fold{fold}_val_preds.txt', val_preds)
            np.savetxt(result_dir+f'/fold{fold}_val_labels.txt', val_labels)
            torch.save(model.state_dict(), result_dir+f'/fold{fold}_model.pth') 
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break








