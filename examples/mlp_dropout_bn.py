import numpy as np

try:
    import cupy as cp
    cp.random.seed(42)
except:
    print("Cupy is not installed!")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

sys.path.append('../pydynet')

import pydynet as pdn
import pydynet.nn.functional as F
import pydynet.nn as nn
from pydynet.optim import Adam
from pydynet.data import data_loader

try:
    import seaborn as sns
    sns.set_theme()
except:
    pass

np.random.seed(42)

data_X, data_y = fetch_olivetti_faces(return_X_y=True)
print(data_X.shape)
train_X, test_X, train_y, test_y = train_test_split(
    data_X,
    data_y,
    train_size=0.8,
)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)


class DNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 40)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DNN_dropout(DNN):

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        return self.fc3(x)


class DNN_BN(DNN):

    def __init__(self) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


use_cuda = True
device = 'cuda' if pdn.cuda.is_available() and use_cuda else 'cpu'

net1 = DNN().to(device)
net2 = DNN_dropout().to(device)
net3 = DNN_BN().to(device)
print(net1)
print(net2)
print(net3)
optim1 = Adam(net1.parameters(), lr=1e-3)
optim2 = Adam(net2.parameters(), lr=1e-3)
optim3 = Adam(net3.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 40

train_loader = data_loader(
    pdn.Tensor(train_X),
    pdn.Tensor(train_y),
    BATCH_SIZE,
    True,
)

train_accs, test_accs = [], []
test_X_cuda = pdn.Tensor(test_X, device=device)
test_y_cuda = pdn.Tensor(test_y, device=device)

bar = tqdm(range(EPOCHES))

from time import time

for epoch in bar:
    # 相同数据训练3个网络
    net1.train()
    net2.train()
    net3.train()

    for batch_X, batch_y in train_loader:
        input_, label = batch_X.to(device), batch_y.to(device)

        output1 = net1(input_)
        l1 = loss(output1, label)
        optim1.zero_grad()
        l1.backward()
        optim1.step()

        output2 = net2(input_)
        l2 = loss(output2, label)
        optim2.zero_grad()
        l2.backward()
        optim2.step()

        output3 = net3(input_)
        l3 = loss(output3, label)
        optim3.zero_grad()
        l3.backward()
        optim3.step()

    net1.eval()
    net2.eval()
    net3.eval()

    # train
    train_right = [0, 0, 0]
    with pdn.no_grad():
        for batch_X, batch_y in train_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            pred1 = net1(input_).argmax(-1)
            pred2 = net2(input_).argmax(-1)
            pred3 = net3(input_).argmax(-1)

            train_right[0] += (pred1.data == label.data).sum().item()
            train_right[1] += (pred2.data == label.data).sum().item()
            train_right[2] += (pred3.data == label.data).sum().item()

        train_acc = np.array(train_right) / len(train_X)

        pred1, pred2, pred3 = (
            net1(test_X_cuda).argmax(-1),
            net2(test_X_cuda).argmax(-1),
            net3(test_X_cuda).argmax(-1),
        )
        test_acc = np.array([
            (pred1.data == test_y_cuda.data).mean().item(),
            (pred2.data == test_y_cuda.data).mean().item(),
            (pred3.data == test_y_cuda.data).mean().item(),
        ])

        bar.set_postfix(
            TRAIN_ACC="{:.3f}, {:.3f}, {:.3f}".format(*train_acc),
            TEST_ACC="{:.3f}, {:.3f}, {:.3f}".format(*test_acc),
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)

plt.plot(
    range(0, 50, 2),
    train_accs[::2, 0],
    label="Train Acc of MLP",
    linewidth=0.7,
    color='blue',
    marker='^',
)
plt.plot(
    range(0, 50, 2),
    test_accs[::2, 0],
    label="Test Acc of MLP",
    linewidth=0.7,
    color='blue',
    marker='*',
)
plt.plot(
    range(0, 50, 2),
    train_accs[::2, 1],
    label="Train Acc of MLP with Dropout",
    linewidth=0.7,
    color='red',
    marker='^',
)
plt.plot(
    range(0, 50, 2),
    test_accs[::2, 1],
    label="Test Acc of MLP with Dropout",
    linewidth=0.7,
    color='red',
    marker='*',
)
plt.plot(
    range(0, 50, 2),
    train_accs[::2, 2],
    label="Train Acc of MLP with BN",
    linewidth=0.7,
    color='orange',
    marker='^',
)
plt.plot(
    range(0, 50, 2),
    test_accs[::2, 2],
    label="Test Acc of MLP with BN",
    linewidth=0.7,
    color='orange',
    marker='*',
)
plt.legend()

plt.savefig("src/dropout_BN.png")
