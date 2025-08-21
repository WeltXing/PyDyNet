import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pydynet as pdn
from pydynet import Tensor
import pydynet.nn as nn
from pydynet.optim import Adam


def windowize(y, input_len, horizon=1, stride=1, step=1):

    y = np.asarray(y)
    max_i = len(y) - (input_len + horizon) * step + step
    idx_inputs = []
    idx_targets = []
    for i in range(0, max_i, stride):
        inp_idx = i + np.arange(0, input_len * step, step)
        tgt_idx = i + input_len * step + np.arange(0, horizon * step, step)
        idx_inputs.append(inp_idx)
        idx_targets.append(tgt_idx)
    X = y[np.array(idx_inputs)]
    Y = y[np.array(idx_targets)]
    return (
        Tensor(X[..., np.newaxis], dtype=np.float32),
        Tensor(Y, dtype=np.float32),
    )


TIME_STEP = 40  # rnn 时序步长数
INPUT_SIZE = 1  # rnn 的输入维度
H_SIZE = 32  # rnn 隐藏单元个数
EPOCHS = 50  # 总共训练次数
h_state = None  # 隐藏层状态


def f(t):
    return np.sin(np.pi * t) + 0.5 * np.cos(2 * np.pi * t)


steps = np.arange(0, 100, .05)
X, Y = windowize(f(steps), input_len=TIME_STEP, horizon=1, stride=1, step=1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
)


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
        _, h_state = self.rnn(x, h_state)
        out = self.out(h_state[:, self.rnn.num_layers - 1, :])
        return out


rnn = RNN()
optimizer = Adam(rnn.parameters(), lr=0.01)
criterion = nn.MSELoss()

loss_list = []

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 0.5

bar = tqdm(range(EPOCHS))
visual_steps = np.arange(0, 10, .05)
visual_X, visual_Y = windowize(f(visual_steps),
                               TIME_STEP,
                               horizon=1,
                               stride=1,
                               step=1)

for step in bar:

    rnn.train()
    prediction = rnn(X_train, h_state)
    train_loss = criterion(prediction, Y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    plt.figure(figsize=(5, 3))
    plt.grid()

    rnn.eval()
    with pdn.no_grad():
        test_loss = criterion(rnn(X_test, h_state), Y_test)

        plt.plot(visual_steps[TIME_STEP:],
                 visual_Y.numpy(),
                 'r-',
                 lw=0.7,
                 label=r'$f(x)=\sin(\pi x)+\cos(2\pi x)/2$')
        plt.plot(
            visual_steps[TIME_STEP:],
            rnn(visual_X, h_state).numpy(),
            'b-.',
            lw=0.7,
            label='Prediction',
        )

    plt.xticks([4, 6, 8, 10])
    plt.yticks([-1.6, -.8, 0, .8])

    plt.legend(loc=1)
    plt.ylim(-1.6, 0.8)
    plt.xlim(visual_steps[TIME_STEP], 10)
    plt.title('Prediction with GRU')
    plt.tight_layout()
    plt.savefig("imgs/rnn.png")
    plt.close()

    bar.set_postfix(
        train_loss="{:.5f}".format(train_loss.item()),
        test_loss="{:.5f}".format(test_loss.item()),
    )
