import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def calculate_descriptors(seqs):
    molecules = [Chem.MolFromFASTA(seq) for seq in seqs]
    descriptors = {
        'BCUT2D_MWLOW': [Descriptors.MolWt(molecule) for molecule in molecules],
        'PEOE_VSA14': [Descriptors.PEOE_VSA14(molecule) for molecule in molecules],
        'SMR_VSA1': [Descriptors.SMR_VSA1(molecule) for molecule in molecules],
        'MinEStateIndex': [Descriptors.MinEStateIndex(molecule) for molecule in molecules],
        'VSA_EState5': [Descriptors.VSA_EState5(molecule) for molecule in molecules],
        'VSA_EState6': [Descriptors.VSA_EState6(molecule) for molecule in molecules],
        'VSA_EState7': [Descriptors.VSA_EState7(molecule) for molecule in molecules],
        'MolLogP': [Descriptors.MolLogP(molecule) for molecule in molecules]
    }
    return pd.DataFrame(descriptors)


def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()



parent_dir = os.path.abspath(os.path.dirname(__file__))
task_name = "ummpep2024"
dataset_dir = "dataset/umami_pep_dataset/"
task_dir = dataset_dir+task_name
result_dir = parent_dir+'/result/'
if not os.path.exists(result_dir): os.mkdir(result_dir)
TASTE_COLUMNS = "Label"

df = pd.read_csv(task_dir+f'/{task_name}_processed.csv', sep='\t')
labels = df[TASTE_COLUMNS].values
features = calculate_descriptors(df['Sequence'].tolist()).values
print(features.shape)


for fold in range(1,6):
    train_ids = load_ids(task_dir+f'/fold{fold}_train_ids.txt')
    test_ids = load_ids(task_dir+f'/fold{fold}_test_ids.txt')
    train_features, test_features = features[train_ids], features[test_ids]
    train_labels, val_labels = labels[train_ids], labels[test_ids]

    model = GradientBoostingClassifier(
        criterion='friedman_mse', 
        loss='log_loss',    
        max_depth=17,  
        min_samples_leaf=3,       
        min_samples_split=10,     
        n_estimators=211        
    )

    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:,1]
   
    np.savetxt(result_dir+f'/fold{fold}_val_preds.txt', preds)
    np.savetxt(result_dir+f'/fold{fold}_val_labels.txt', val_labels)
    joblib.dump(model, result_dir + f'/fold{fold}_model.pkl')







