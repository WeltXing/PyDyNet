from os.path import join
from tqdm import tqdm

import pydynet as pdn
import pydynet.nn as nn
import pydynet.nn.functional as F
from pydynet.optim import Adam
from pydynet.data import data_loader

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

path = r'examples/data/CoLA/tokenized'


def extract(line: str):
    lines = line.split('\t')
    y = int(lines[1])
    sentence = lines[-1][:-1]
    return sentence.split(), y


def load_data():

    with open(join(path, 'in_domain_train.tsv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sens, ys = [], []
    max_len = -1
    word_dict = set()
    for line in tqdm(lines):
        x, y = extract(line)
        word_dict = word_dict.union(set(x))
        max_len = max(max_len, len(x))
        sens.append(x)
        ys.append(y)
    word_dict = list(word_dict)

    X = np.zeros((len(lines), max_len), dtype=int)
    for i in tqdm(range(len(lines))):
        for j, word in enumerate(sens[i]):
            X[i, j] = word_dict.index(word) + 1
    y = np.array(ys)

    return X, y


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size
                ), "Embedding size needs to be divisible by heads"

        self.Q = nn.Linear(self.embed_size,
                           self.embed_size,
                           bias=False,
                           dtype=np.float32)
        self.K = nn.Linear(self.embed_size,
                           self.embed_size,
                           bias=False,
                           dtype=np.float32)
        self.V = nn.Linear(self.embed_size,
                           self.embed_size,
                           bias=False,
                           dtype=np.float32)
        self.O = nn.Linear(self.embed_size,
                           self.embed_size,
                           bias=False,
                           dtype=np.float32)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], query.shape[1]

        xq, xk, xv = (
            self.Q(values).reshape(N, value_len, self.heads, self.head_dim),
            self.K(values).reshape(N, key_len, self.heads, self.head_dim),
            self.V(values).reshape(N, query_len, self.heads, self.head_dim),
        )

        # Split the embedding into self.heads different pieces
        xq, xkT = xq.transpose(0, 2, 1, 3), xk.transpose(0, 2, 3, 1)
        attention = xq @ xkT / self.head_dim**.5

        if mask is not None:
            mask[mask.eq(1)] = np.float32('-inf')
            attention = attention + mask

        attention = F.softmax(attention, axis=-1)
        output = attention @ xv.transpose(0, 2, 1, 3)

        output = output.transpose(0, 2, 1, 3).reshape(N, value_len, -1)
        return self.O(output)


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size, dtype=np.float32)
        self.norm2 = nn.LayerNorm(embed_size, dtype=np.float32)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,
                      forward_expansion * embed_size,
                      dtype=np.float32),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size,
                      embed_size,
                      dtype=np.float32),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = (self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = (self.norm2(forward + x))
        return out


def sinusoidal_positional_encoding(max_len: int, d_model: int):
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pdn.Tensor(pe.astype(np.float32))


@pdn.no_grad()
def construct_mask(x: pdn.Tensor, padding_idx=0):
    mask = x.eq(padding_idx)  # [batch_size, seq_len]
    return pdn.unsqueeze(mask, (1, 2)).astype(
        np.float32)  # [batch_size, 1, 1, seq_len]


class Transformer(nn.Module):

    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        vocab_size,
        max_length,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0,
            dtype=np.float32,
        )
        self.position_embedding = nn.Parameter(
            sinusoidal_positional_encoding(max_length, embed_size), False)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            ) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, 1, dtype=np.float32)

    def forward(self, x, mask):
        a = self.word_embedding(x)
        out = a + self.position_embedding

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = out[:, 0, :]
        return self.fc_out(out)


if __name__ == "__main__":
    LR = 5e-4
    EPOCHES = 100
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 512
    use_cuda = True

    device = 'cuda' if pdn.cuda.is_available() and use_cuda else 'cpu'

    X, y = load_data()
    y[y == 0] = -1

    train_X, test_X, train_y, test_y = train_test_split(
        pdn.Tensor(X),
        pdn.Tensor(y),
        train_size=0.8,
        stratify=y,
        shuffle=True,
    )

    ratio_pos = (train_y.mean() + 1) / 2

    train_loader = data_loader(
        train_X,
        train_y,
        shuffle=False,
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = data_loader(
        test_X,
        test_y,
        shuffle=False,
        batch_size=TEST_BATCH_SIZE,
    )

    net = Transformer(512, 1, 4, 3, 0.05, X.max() + 1, 44).to(device)
    optimizer = Adam(net.parameters(), lr=LR)
    bar = tqdm(range(EPOCHES))
    info_list = []
    for epoch in bar:

        net.train()

        for batch_X, batch_y in train_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            output = net(input_, construct_mask(input_))
            weight = pdn.ones(label.shape, dtype=np.float32)
            weight[label == -1] = 1 / (1 - ratio_pos)
            weight[label == 1] = 1 / ratio_pos
            loss = (weight.to(device) *
                    pdn.log(1 + pdn.exp(-label * pdn.squeeze(output)))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        train_right, train_size = 0, 0
        test_right, test_size = 0, 0

        with pdn.no_grad():
            for batch_X, batch_y in train_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = pdn.sign(
                    pdn.squeeze(net(input_, construct_mask(input_))))
                train_right += (pred.data == label.data).sum()
                train_size += batch_X.shape[0]

            for batch_X, batch_y in test_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = pdn.sign(
                    pdn.squeeze(net(input_, construct_mask(input_))))
                test_right += (pred.data == label.data).sum()
                test_size += batch_X.shape[0]

        train_acc, test_acc = train_right / train_size, test_right / test_size
        bar.set_postfix(
            Loss="{:.6f}".format(loss.item()),
            TEST_ACC="{:.4f}".format(test_acc),
            TRAIN_ACC="{:.4f}".format(train_acc),
        )
        info_list.append([train_acc.item(), test_acc.item()])

    info_list = np.array(info_list)

    plt.figure(figsize=(5, 3))

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 0.5

    plt.grid(zorder=-10)
    plot_kwargs = {'linewidth': 0.7, 'zorder': 10}

    x = np.arange(0, 100, 4) + 2
    plt.plot(x,
             info_list[::4, 0],
             label="Training accuracy",
             color='blue',
             marker='^',
             **plot_kwargs,
             linestyle='-')
    plt.plot(x,
             info_list[::4, 1],
             label="Test accuracy",
             color='red',
             marker='*',
             **plot_kwargs,
             linestyle='--')

    plt.xlim(0, 100)
    plt.ylim(.4, 1)

    plt.yticks([.4, .6, .8, 1], size=13)
    plt.xticks([20, 40, 60, 80, 100], size=13)
    plt.xlabel("Epochs", size=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/transformer.png")
