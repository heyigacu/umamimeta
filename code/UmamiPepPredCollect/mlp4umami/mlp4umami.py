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
# from sklearn.preprocessing import StandardScaler

parent_dir = os.path.abspath(os.path.dirname(__file__))
task_name = "ummpep2024"
dataset_dir = "dataset/umami_pep_dataset/"
task_dir = dataset_dir+task_name
result_dir = parent_dir+'/result/'
if not os.path.exists(result_dir): os.mkdir(result_dir)

def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()


def df_column_slice(df, column, id_list):
    new_df = pd.DataFrame() 
    for i in id_list:
        new_df = pd.concat([new_df, df.loc[df[column] == i]])
    return new_df

def get_subset_dataloader(dataset, ids, batch_size=32, shuffle=False):
    indices = [i for i, sample_id in enumerate(dataset.all_ids) if sample_id in ids]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_molecular_fingerprints(seqs_list):
    fingerprints = []
    for seq in seqs_list:
        mol = Chem.MolFromFASTA(seq)
        if mol:
            fp = RDKFingerprint(mol)
            fingerprints.append(np.array(fp))  
        else:
            fingerprints.append(np.zeros(2048)) 
    return np.array(fingerprints)

TASTE_COLUMNS = 'Label'

class PepDataset(Dataset):
    def __init__(self, csv_file, ids_file):
        data = pd.read_csv(csv_file, sep='\t')
        labels = data[TASTE_COLUMNS].values
        feature_embeddings = get_molecular_fingerprints(data['Sequence'].tolist())
        self.selected_ids = load_ids(ids_file)
        self.features = feature_embeddings[self.selected_ids]
        self.labels = labels[self.selected_ids]
        print(self.selected_ids[0])
        self.feature_dim = self.features.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # Change to 1 for binary classification
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x) 


for fold in range(1,6):

    train_dataset = PepDataset(task_dir+f'/{task_name}_processed.csv', task_dir+f'/fold{fold}_train_ids.txt')
    input_size = train_dataset.feature_dim
    test_dataset = PepDataset(task_dir+f'/{task_name}_processed.csv', task_dir+f'/fold{fold}_test_ids.txt')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLP(input_size)
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
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze() 
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
            for features, labels in val_loader:
                outputs = model(features).squeeze() 
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