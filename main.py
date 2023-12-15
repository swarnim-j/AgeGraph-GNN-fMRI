from AgeGraph.datasets import AgeDataset
import argparse
import torch
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os, random
import os.path as osp
import matplotlib.pyplot as plt
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPAge')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--hidden_mlp', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--echo_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
path = "base_params/"
res_path = "results/"
root = "data/"

'''
class ArgsClass(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = ArgsClass()
args.dataset = 'HCPAge'
args.runs = 1
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.hidden = 32
args.hidden_mlp = 64
args.num_layers = 3
args.epochs = 50
args.echo_epoch = 50
args.batch_size = 16
args.early_stopping = 50
args.lr = 1e-5
args.weight_decay = 0.0005
args.dropout = 0.5
'''

if not osp.isdir(path):
    os.mkdir(path)
if not osp.isdir(res_path):
    os.mkdir(res_path)

def logger(info):
    f = open(osp.join(res_path, 'results_new.csv'), 'a')
    print(info, file=f)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

dataset = AgeDataset(root=root)

print("dataset loaded successfully!", args.dataset)
labels = [data.y.item() for data in dataset]

train_temp, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=123, shuffle=True)
temp = dataset[train_temp]
train_labels = [data.y.item() for data in temp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))), test_size=0.125, stratify=train_labels, random_state=123, shuffle = True)

train_dataset = temp[train_indices]
val_dataset = temp[val_indices]
test_dataset = dataset[test_indices]

print(f"dataset {args.dataset} loaded with train {len(train_dataset)} val {len(val_dataset)} test {len(test_dataset)} splits")

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

args.num_features, args.num_classes = dataset.num_features, dataset.num_classes

def train(train_loader):
    model.train()
    total_loss = 0
    train_loader.num_nodes = 1000

    for data in train_loader:
        print(data)
        if data.size()[0] < 16000:
            data.batch = torch.cat([data.batch, torch.zeros(16000 - data.size()[0], dtype=torch.long)])

        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data)

        loss = critereon(out, data.y)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    loader.num_nodes = 1000
    correct = 0

    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)

model = GraphNetwork(dataset=train_dataset, hidden_channels=args.hidden).to(args.device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
critereon = torch.nn.CrossEntropyLoss()

loss, test_acc = [], []
best_val_acc, best_val_loss = 0.0, 0.0
val_acc_history, test_acc_history, test_loss_history = [], [], []

losses = []

for epoch in range(args.epochs):
    loss = train(train_loader)
    test_acc = test(test_loader)
    val_acc = test(val_loader)

    print(f"epoch: {epoch}, loss: {np.round(loss.item(), 6)}, val_acc:{np.round(val_acc, 2)}, test_acc:{np.round(test_acc, 2)}")
    val_acc_history.append(val_acc)
    test_acc_history.append(test_acc)

    losses.append(loss.item())

plt.plot(losses)
plt.show()

log = f"dataset, {args.dataset}, model, {args.model}, hidden, {args.hidden}, epochs, {args.epochs}, batch size, {args.batch_size}, loss, {round(np.mean(losses), 4)}, acc, {round(np.mean(test_acc_history) * 100, 2)}, std, {round(np.std(test_acc_history) * 100, 2)}"
print(log)

log = f"{args.dataset}, {args.model}, {args.hidden}, {args.epochs}, {args.batch_size}, {round(np.mean(losses), 4)}, {round(np.mean(test_acc_history) * 100, 2)}, {round(np.std(test_acc_history) * 100, 2)}"
logger(log)