import sys

sys.path.append('../pydynet')

import pydynet as pdn
from pydynet.tensor import Tensor
import pydynet.nn as nn
from pydynet.optim import Adam

import numpy as np
try:
    import cupy as cp
    cp.random.seed(42)
except:
    print("Cupy is not installed!")

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme()
except:
    pass

np.random.seed(42)

device = 'cuda' if pdn.cuda.is_available() else 'cpu'

TIME_STEP = 41  # rnn 时序步长数
INPUT_SIZE = 1  # rnn 的输入维度
H_SIZE = 64  # of rnn 隐藏单元个数
EPOCHS = 150  # 总共训练次数
h_state = None  # 隐藏层状态


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=H_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(H_SIZE, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.reshape(-1, H_SIZE)
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN().to(device)
optimizer = Adam(rnn.parameters(), lr=0.005)
criterion = nn.MSELoss()

loss_list = []

rnn.train()
for step in range(EPOCHS):
    start, end = 2 * step * np.pi, 2 * (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(2 * steps)
    x = Tensor(x_np[np.newaxis, :, np.newaxis]).to(device)
    y = Tensor(y_np[np.newaxis, :, np.newaxis]).to(device)
    prediction, h_state = rnn(x, h_state)  #
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.xticks(
        [start, (start + end) / 2, end],
        [r"${}\pi$".format(x) for x in range(2 * step, 2 * step + 3)],
    )
    plt.plot(steps, y_np.flatten(), 'r-', lw=0.7, marker='*', label='target')
    plt.plot(steps,
             prediction.numpy().flatten(),
             'b-',
             lw=0.7,
             marker='^',
             label='prediction')
    plt.legend()
    plt.ylim(-1.1, 1.1)
    plt.savefig("src/rnn.png")
    plt.close()
