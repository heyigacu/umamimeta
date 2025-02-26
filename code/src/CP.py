import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split 
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pickle as pk
from functools import lru_cache
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np

class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)

DISTANCE_METRICS = {
    "Cosine": Cosine,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()


def df_column_slice(df, column, id_list):
    new_df = pd.DataFrame() 
    for i in id_list:
        new_df = pd.concat([new_df, df.loc[df[column] == i]])
    return new_df

class ProPepDataset(Dataset):
    def __init__(self, csv_file, ids_file):
        data = pd.read_csv(csv_file, sep='\t')
        labels = data['Label'].values
        total_feature_embeddings = np.loadtxt(feature_path)
        pro_features = total_feature_embeddings[list(data['Protein_idx'])]
        pep_features = total_feature_embeddings[list(data['Peptide_idx'])]
        self.selected_ids = load_ids(ids_file)
        self.pro_features = pro_features[self.selected_ids]
        self.pep_features = pep_features[self.selected_ids]
        self.labels = labels[self.selected_ids]
        self.feature_dim = self.pro_features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pro_feature = torch.tensor(self.pro_features[idx], dtype=torch.float32)
        pep_feature = torch.tensor(self.pep_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pro_feature, pep_feature, label

class PredictDataset(Dataset):
    def __init__(self, csv_file, ids_file):
        data = pd.read_csv(csv_file, sep='\t')
        labels = data['Label'].values
        total_feature_embeddings = np.loadtxt(feature_path)
        pro_features = total_feature_embeddings[list(data['Protein_idx'])]
        pep_features = total_feature_embeddings[list(data['Peptide_idx'])]
        self.selected_ids = load_ids(ids_file)
        self.pro_features = pro_features[self.selected_ids]
        self.pep_features = pep_features[self.selected_ids]
        self.labels = labels[self.selected_ids]


        self.feature_dim = self.pro_features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pro_feature = torch.tensor(self.pro_features[idx], dtype=torch.float32)
        pep_feature = torch.tensor(self.pep_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pro_feature, pep_feature, label


# parent_dir = os.path.abspath(os.path.dirname(__file__))
# parent_parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# task_name = "Kd"
# ESM_name = "esm2_t6_8M_UR50D" #esm1v_t33_650M_UR90S_1
# dataset_dir = "dataset/public_pep_dataset/"
# task_dir = dataset_dir+task_name
# feature_path = task_dir+f'/{ESM_name}.txt'
# result_dir = parent_parent_dir+f'/result/{task_name}_{ESM_name}/'
# if not os.path.exists(result_dir): os.makedirs(result_dir)

# parent_dir = os.path.abspath(os.path.dirname(__file__))
# parent_parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# task_name = "IC50"
# ESM_name = "esm2_t6_8M_UR50D" #esm1v_t33_650M_UR90S_1
# dataset_dir = "dataset/public_pep_dataset/"
# task_dir = dataset_dir+task_name
# feature_path = task_dir+f'/{ESM_name}.txt'
# result_dir = parent_parent_dir+f'/result/{task_name}_{ESM_name}/'
# if not os.path.exists(result_dir): os.makedirs(result_dir)

job = "bi_classify"
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
task_name = "Kd"
ESM_name = "esm2_t6_8M_UR50D" #esm1v_t33_650M_UR90S_1
dataset_dir = "dataset/public_pep_dataset/"
task_dir = dataset_dir+task_name
feature_path = task_dir+f'/{ESM_name}.txt'
result_dir = parent_parent_dir+f'/result/{task_name}_{ESM_name}/'
if not os.path.exists(result_dir): os.makedirs(result_dir)

def train():
    for fold in range(1,6):
        train_dataset = ProPepDataset(task_dir+f'/{task_name}_processed.csv', task_dir+f'/fold{fold}_train_ids.txt')
        input_size = train_dataset.feature_dim
        test_dataset = ProPepDataset(task_dir+f'/{task_name}_processed.csv', task_dir+f'/fold{fold}_test_ids.txt')
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = SimpleCoembeddingSigmoid(input_size, input_size)
        if job == "regress":
            criterion = nn.MSELoss()
        elif job == "bi_classify":
            criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 100
        # Early stopping variables
        patience = 8
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for pro_features, pep_features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(pro_features, pep_features).squeeze() 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            val_loss = 0.0
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for pro_features, pep_features, labels in val_loader:
                    outputs = model(pro_features, pep_features).squeeze() 
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_preds.append(outputs)
                    val_labels.append(labels)

            avg_val_loss = val_loss / len(val_loader)
            val_preds = torch.cat(val_preds).numpy()
            val_labels = torch.cat(val_labels).numpy()
            print(f'epoch:{epoch}, train_loss:{train_loss}, val_loss:{val_loss}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                np.savetxt(result_dir+f'/fold{fold}_val_preds.txt', val_preds)
                np.savetxt(result_dir+f'/fold{fold}_val_labels.txt', val_labels)
                torch.save(model.state_dict(), result_dir+f'/fold{fold}_model.pth') 
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

def predict(pro_features, pep_features):
    outputs = 0.
    for fold in range(1,6):
        input_size = pro_features.shape[1]
        model = SimpleCoembeddingSigmoid(input_size, input_size, classify=False)
        model.load_state_dict(torch.load(result_dir + f'/fold{fold}_model.pth', map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():

            outputs += model(pro_features.float(), pep_features.float()).squeeze() 


    return outputs/fold
    
# train()
esm_model = "esm2_t6_8M_UR50D"
rec_dataset_dir = "dataset/umami_rec_dataset"
rec_taskname = 'ummrec'
lig_dataset_dir = "dataset/umami_pep_dataset"
lig_taskname = 'ummpep2024'
df_lig = pd.read_csv(f"{lig_dataset_dir}/{lig_taskname}/{lig_taskname}_processed.csv", sep="\t")
df_rec = pd.read_csv(f"{rec_dataset_dir}/{rec_taskname}/{rec_taskname}_processed.csv", sep="\t")
lig_embeddings = np.loadtxt(f"{lig_dataset_dir}/{lig_taskname}/esm2_t6_8M_UR50D.txt")
rec_embeddings = np.loadtxt(f"{rec_dataset_dir}/{rec_taskname}/esm2_t6_8M_UR50D.txt")

# feature = np.loadtxt("dataset/public_pep_dataset/IC50/esm2_t6_8M_UR50D.txt")
# df = pd.read_csv("dataset/public_pep_dataset/IC50/IC50_processed.csv", sep='\t')
# rec_embeddings = feature[list(df['Protein_idx'])]
# lig_embeddings = feature[list(df['Peptide_idx'])]

ls = []
for i in range(lig_embeddings.shape[0]):
    print(i)
    temp_ls = []
    for j in range(rec_embeddings.shape[0]):
        rst = predict(torch.from_numpy(np.array([rec_embeddings[j]])), torch.from_numpy(np.array([lig_embeddings[i]])))
        temp_ls.append(rst.numpy())
    ls.append(temp_ls)
df = pd.DataFrame(np.array(ls))
df.index = list(df_lig['Sequence'])
df.columns = list(df_rec['pocket'])
df.to_csv('Kd.csv', sep='\t')