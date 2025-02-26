import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import joblib
from sklearn.model_selection import KFold,StratifiedKFold

parent_dir = os.path.abspath(os.path.dirname(__file__))

result_dir = parent_dir+'/result/'
if not os.path.exists(result_dir): os.mkdir(result_dir)

def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()




df = pd.read_csv(parent_dir+f'/ummpep2024_processed_2-15.csv', sep='\t', index_col=0)
labels = df["Label"].values
ids = df["ID"].values
indexs = list(range(len(ids)))
dic = dict(zip(ids, indexs))
features = pd.read_csv(parent_dir+f'/encoding.tsv', sep='\t', index_col=0).values
task_dir = "dataset/umami_pep_dataset/ummpep2024/"





for fold in range(1,6):
    train_ids = load_ids(task_dir+f'/fold{fold}_train_ids.txt')
    train_ids = [x for x in train_ids if x in ids]
    train_indexs = [dic[id] for id in train_ids]

    test_ids = load_ids(task_dir+f'/fold{fold}_test_ids.txt')
    test_ids = [x for x in test_ids if x in ids]
    test_indexs = [dic[id] for id in test_ids]

    train_features, train_labels  = features[train_indexs], labels[train_indexs]

    test_features, test_labels = features[test_indexs], labels[test_indexs]
    

    model = RandomForestClassifier(random_state=42)
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    np.savetxt(result_dir + f'/fold{fold}_val_preds.txt', preds)
    np.savetxt(result_dir + f'/fold{fold}_val_labels.txt', test_labels)
    joblib.dump(model, result_dir + f'/fold{fold}_model.pkl')

