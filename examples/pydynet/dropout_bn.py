import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import pydynet as pdn
import pydynet.nn.functional as F
import pydynet.nn as nn
from pydynet.optim import Adam
from pydynet.data import data_loader

data_X, data_y = fetch_olivetti_faces(return_X_y=True)
print(data_X.shape)
train_X, test_X, train_y, test_y = train_test_split(
    data_X,
    data_y,
    train_size=0.8,
    stratify=data_y,
    random_state=42,
)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)


class DNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4096, 512, dtype=np.float32)
        self.fc2 = nn.Linear(512, 128, dtype=np.float32)
        self.fc3 = nn.Linear(128, 40, dtype=np.float32)

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
        self.bn1 = nn.BatchNorm1d(512, dtype=np.float32)
        self.bn2 = nn.BatchNorm1d(128, dtype=np.float32)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


np.random.seed(42)
use_cuda = True
device = f'cuda:{pdn.cuda.device_count() - 1}' if pdn.cuda.is_available(
) else 'cpu'

net1 = DNN().to(device)
net2 = DNN_dropout().to(device)
net3 = DNN_BN().to(device)
print(net1)
print(net2)
print(net3)
optim1 = Adam(net1.parameters(), lr=5e-5)
optim2 = Adam(net2.parameters(), lr=5e-5)
optim3 = Adam(net3.parameters(), lr=5e-5)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 40

train_loader = data_loader(pdn.Tensor(train_X), pdn.Tensor(train_y),
                           BATCH_SIZE, True)

train_accs, test_accs = [], []
test_X_cuda = pdn.Tensor(test_X, device=device)
test_y_cuda = pdn.Tensor(test_y, device=device)

bar = tqdm(range(EPOCHES))

for epoch in bar:
    # 相同数据训练3个网络
    net1.train()
    net2.train()
    net3.train()

    for batch_X, batch_y in train_loader:
        input_, label = batch_X.to(device), batch_y.to(device)

        output1 = net1(input_)
        l1 = loss(output1, label)
        output2 = net2(input_)
        l2 = loss(output2, label)
        output3 = net3(input_)
        l3 = loss(output3, label)

        optim1.zero_grad()
        optim2.zero_grad()
        optim3.zero_grad()
        (l1 + l2 + l3).backward()
        optim1.step()
        optim2.step()
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

            train_right[0] += pred1.eq(label).sum().item()
            train_right[1] += pred2.eq(label).sum().item()
            train_right[2] += pred3.eq(label).sum().item()

        train_acc = np.array(train_right) / len(train_X)

        pred1, pred2, pred3 = (
            net1(test_X_cuda).argmax(-1),
            net2(test_X_cuda).argmax(-1),
            net3(test_X_cuda).argmax(-1),
        )
        test_acc = np.array([
            pred1.eq(test_y_cuda.data).mean().item(),
            pred2.eq(test_y_cuda.data).mean().item(),
            pred3.eq(test_y_cuda.data).mean().item(),
        ])

        bar.set_postfix(
            TRAIN_ACC="{:.3f}, {:.3f}, {:.3f}".format(*train_acc),
            TEST_ACC="{:.3f}, {:.3f}, {:.3f}".format(*test_acc),
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)

plt.figure(figsize=(9, 3))

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 0.5

plot_kwargs = {'linewidth': 0.7, 'zorder': 10}

plt.subplot(1, 2, 1)
plt.grid(zorder=-10)

plt.xlim(2, 50)
plt.ylim(0, 1.05)

x = np.arange(0, 50, 2) + 2
plt.plot(x,
         train_accs[::2, 0],
         label="MLP",
         color='blue',
         marker='^',
         **plot_kwargs)
plt.plot(x,
         train_accs[::2, 1],
         label="MLP with Dropout",
         color='green',
         marker='s',
         **plot_kwargs)
plt.plot(x,
         train_accs[::2, 2],
         label="MLP with BN",
         color='red',
         marker='*',
         **plot_kwargs)

plt.yticks([0, .2, .4, .6, .8, 1], size=13)
plt.xticks([10, 20, 30, 40, 50], size=13)
plt.xlabel("Epochs", size=13)
plt.title("Training Accuracy on Olivetti Faces Dataset")
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.grid(zorder=-10)

plt.xlim(2, 50)
plt.ylim(0, 1.)

plt.plot(x,
         test_accs[::2, 0],
         label="MLP",
         color='blue',
         marker='^',
         **plot_kwargs)
plt.plot(x,
         test_accs[::2, 1],
         label="MLP with Dropout",
         color='green',
         marker='s',
         **plot_kwargs)
plt.plot(x,
         test_accs[::2, 2],
         label="MLP with BN",
         color='red',
         marker='*',
         **plot_kwargs)

plt.yticks([0, .2, .4, .6, .8, 1], size=13)
plt.xticks([10, 20, 30, 40, 50], size=13)
plt.xlabel("Epochs", size=13)
plt.title("Test Accuracy on Olivetti Faces Dataset")
plt.legend()
plt.tight_layout()

plt.savefig("imgs/dropout_bn.png")
