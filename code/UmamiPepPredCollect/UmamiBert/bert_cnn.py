
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig, RobertaConfig, RobertaModel, AlbertConfig
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
parent_dir = os.path.abspath(os.path.dirname(__file__))



class SimpleTokenizer:
    def __init__(self, max_length=16):
        self.tokens = ['[PAD]', '[CLS]', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                       'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.max_length = max_length

    def encode(self, text):
        tokens = list(text)
        input_ids = [self.token_to_id['[CLS]']] 
        input_ids += [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        if len(input_ids) < self.max_length:
            input_ids += [self.token_to_id['[PAD]']] * (self.max_length - len(input_ids))
        attention_masks = [1] * len(input_ids)
        attention_masks[len(tokens)+1:] = [0] * (self.max_length - len(tokens) - 1)
        return input_ids, attention_masks

    # def encode_seqs(self, seqs):
    #     input_idss = []
    #     attention_maskss = []
    #     for seq in seqs:
    #         input_ids,attention_masks = self.encode(seq)
    #         input_idss.append(input_ids)
    #         attention_maskss.append(attention_masks)
    #     return torch.from_numpy(np.array(input_idss)), torch.from_numpy(np.array(attention_maskss))

    def decode(self, input_ids):
        tokens = [self.id_to_token[id_] for id_ in input_ids if id_ != self.token_to_id['[PAD]']] 
        return ''.join(tokens[1:])



class BertCNNModel(nn.Module):
    def  __init__(self):
        super(BertCNNModel, self).__init__()
        self.num_labels = 1
        self.hidden_size = 768
        model_config = BertConfig(
                hidden_size=self.hidden_size,
                num_attention_heads=8,
                num_hidden_layers=8,
                intermediate_size=2048,
                max_position_embeddings=512,
                vocab_size=22,
            )
        self.window_sizes = [1,2,3]
        self.max_text_len = 16  #self.num_labels + 2
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(0.1)
        self.dropout_rate = 0.1
        self.filter_size = 250
        self.dense_1 = nn.Linear(self.hidden_size, 1)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                    out_channels=self.filter_size,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=self.filter_size * len(self.window_sizes),
                            out_features=self.num_labels)

    def forward(self, inputs, token_type_ids=None, attention_mask=None, position_ids=None):
        outputs = self.bert(inputs, attention_mask=attention_mask)
        embed_x = outputs[0]
        embed_x = self.dropout(embed_x)

        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out).squeeze(1)
        out = out.sigmoid()

        return out


def load_ids(file_path):
    return pd.read_csv(file_path,header=None).values.squeeze()


class TextDataset(Dataset):
    def __init__(self, csv_file, ids_file, tokenizer, max_length=16):
        data = pd.read_csv(csv_file, sep='\t')
        labels = data['Label'].values
        seqs = data['Sequence'].to_list()
        self.selected_ids = load_ids(ids_file)
        self.labels = labels[self.selected_ids]
        self.texts = [seqs[id] for id in self.selected_ids]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids, attention_mask = self.tokenizer.encode(text)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label, dtype=torch.float)





tokenizer = SimpleTokenizer()
task_dir = "dataset/umami_pep_dataset/ummpep2024/"
result_dir = parent_dir+'/result/'
if not os.path.exists(result_dir): os.mkdir(result_dir)

for fold in range(1,6):
    train_dataset = TextDataset(task_dir+f'/ummpep2024_processed.csv', task_dir+f'/fold{fold}_train_ids.txt', tokenizer)
    test_dataset = TextDataset(task_dir+f'/ummpep2024_processed.csv', task_dir+f'/fold{fold}_test_ids.txt', tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BertCNNModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    # Early stopping variables
    patience = 8
    best_val_loss = float('inf')
    counter = 0

    best_preds = []
    best_labels = []


    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask).squeeze() 
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
            for input_ids, attention_mask, labels in val_loader:
                outputs = model(input_ids, attention_mask=attention_mask).squeeze() 
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

