# %%
import os
import pickle
import torch
import pandas as pd

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from datasets import *


# %%
data_folder = 'data/ClassifyAppDataset'
num_epochs = 100
batch_size = 64
dense_layer_size = 200
num_layer = 2
print_summary = False
out_folder = 'output/inst2vec_for_classifyapp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_step = 10
max_length = 512
emb_path = 'src/observation/inst2vec/pickle/embeddings.pickle'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# %% [markdown]
# ## 加载数据集

# %%
def collate_fn(batch, padding_value=8564, max_length=max_length):
    input_ids, labels = [item['input_ids'] for item in batch], [item['labels'] for item in batch]
    padded_batch = []
    if max_length == None:
        max_length = max(len(item) for item in input_ids)
    
    for item in input_ids:
        padded_item = item + [padding_value] * max(0, (max_length - len(item)))
        padded_item = padded_item[:max_length]
        padded_batch.append(padded_item)
    return {"input_ids": torch.tensor(padded_batch), "labels": torch.tensor(labels)}

dataset = load_from_disk("/root/Compiler-master/data/ClassifyAppDataset")
train_loader = DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(dataset['val'], batch_size=batch_size, collate_fn=collate_fn)

# %%
# 定义网络结构
class ClassifyAppLSTM(nn.Module):
    def __init__(self, embedding_dim, dense_layer_size, num_classes, num_layers, dropout):
        super(ClassifyAppLSTM, self).__init__()
        # Embedding 
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embedding_matrix_normalized = F.normalize(embeddings, p=2, dim=1)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix_normalized, freeze=False)

        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(embedding_dim * 2, num_classes)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(hidden_dim, 1),
        #     torch.nn.Sigmoid()
        # )

        # Dense layers
        # self.dense1 = nn.Linear(embedding_dim * 2, dense_layer_size)
        # self.dense2 = nn.Linear(dense_layer_size, num_classes)

        # Activation functions
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # LSTM layers
        x, _ = self.lstm(x)

        # Take the output of the last time step
        # x = 

        # Dense layers
        # x = self.relu(self.dense1(x))
        # x = self.dense2(x)
        return self.fc(x[:, -1, :])

        # return x
    

model = ClassifyAppLSTM(200, 200, 104, 3, 0.5)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
def eval_model(model, loader):
    model.eval()
    correct = 0
    y_true, y_pred = [], []
    progress_bar = tqdm(loader, desc='Eval', leave=False)
    with torch.no_grad():
        for idx, batch in enumerate(progress_bar):
            data = {k: v.to(device) for k, v in batch.items()}
            outputs = model(data['input_ids'])
            preds = outputs.argmax(dim=1)
            y_pred += preds.tolist()
            y_true += data['labels'].tolist()
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    progress_bar.set_postfix(f1=f1_weighted, acc=acc)
    return f1_weighted, acc

def train_model(model, train_loader, val_loader,  criterion, optimizer, num_epochs):
    # 模型训练
    writer = SummaryWriter(out_folder)
    pre_val_f1 = 0
    gloabl_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        step = 0
        model.train()
        y_true, y_pred = [], []
        acc_num = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        running_loss = 0.0
        for idx, batch in enumerate(progress_bar):
            data = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(data['input_ids'])
            loss = criterion(outputs, data['labels'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            y_pred += preds.tolist()
            y_true += data['labels'].tolist()

            if gloabl_step % log_step == 0:
                writer.add_scalar('train_loss', loss.item(), gloabl_step)
            gloabl_step += 1

            running_loss += loss.item()
            acc_num += torch.sum(data['labels'] == preds).item()
            progress_bar.set_postfix(loss=running_loss / (idx + 1), acc=acc_num / len(y_pred))
                    
        train_f1, train_acc = f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred)
        val_f1, val_acc = eval_model(model, val_loader)
        writer.add_scalar('train_f1', train_f1, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_f1', val_f1, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        if val_f1 > pre_val_f1:
            pre_val_f1 = val_f1
            torch.save(model.state_dict(), out_folder + f'/best_epoch_{epoch}_eval_f1_{int(val_f1*100)}_acc_{int(val_acc*100)}.pth')
            
        torch.save(model.state_dict(), out_folder + f'/best_epoch_{epoch}_eval_f1_{int(val_f1*100)}_acc_{int(val_acc*100)}.pth')
        


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)


