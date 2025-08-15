import sys

sys.path.append('../pydynet')

import pydynet as pdn
from pydynet.tensor import Tensor
import pydynet.nn as nn
from pydynet.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

device = 'cuda' if pdn.cuda.is_available() else 'cpu'

TIME_STEP = 41  # rnn 时序步长数
INPUT_SIZE = 1  # rnn 的输入维度
H_SIZE = 128  # rnn 隐藏单元个数
EPOCHS = 200  # 总共训练次数
h_state = None  # 隐藏层状态


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=H_SIZE,
            num_layers=1,
            batch_first=True,
            dtype=np.float32,
        )
        self.out = nn.Linear(H_SIZE, 1, dtype=np.float32)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.reshape(-1, H_SIZE)
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN().to(device)
optimizer = Adam(rnn.parameters(), lr=0.01)
criterion = nn.MSELoss()

loss_list = []

rnn.train()

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 0.5

bar = tqdm(range(EPOCHS))

for step in bar:
    start, end = 2 * step * np.pi, 2 * (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.sin(np.sin(steps) + 0.4 * np.cos(3 * steps))
    x = Tensor(x_np[np.newaxis, :, np.newaxis], dtype=np.float32).to(device)
    y = Tensor(y_np[np.newaxis, :, np.newaxis], dtype=np.float32).to(device)
    prediction, h_state = rnn(x, h_state)  #
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.figure(figsize=(5, 3))

    plt.xticks(
        [start, (start + end) / 2, end],
        [r"${}\pi$".format(x) for x in range(2 * step, 2 * step + 3)],
    )
    plt.yticks([-1, -.5, 0, .5, 1])
    plt.plot(steps,
             y_np.flatten(),
             'r-',
             lw=0.7,
             marker='*',
             label=r'$f(x)=\sin(\sin(x)+0.4\cos(3x))$')
    plt.plot(steps,
             prediction.numpy().flatten(),
             'b-',
             lw=0.7,
             marker='^',
             label='prediction')
    plt.legend()
    plt.ylim(-1.1, 1.1)
    plt.xlim(start, end)
    plt.title('Prediction with GRU')
    plt.tight_layout()
    plt.savefig("imgs/rnn.png")
    plt.close()

    bar.set_postfix(Loss="{:.5f}".format(loss.item()))
