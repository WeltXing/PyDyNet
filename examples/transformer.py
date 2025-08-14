import sys

sys.path.append('../pydynet')

from os.path import join
from tqdm import tqdm

import pydynet as pdn
from pydynet.tensor import Tensor
import pydynet.nn as nn
import pydynet.nn.functional as F
from pydynet.optim import Adam
from pydynet.data import data_loader

import numpy as np
try:
    import cupy as cp
except:
    cp.random.seed(42)
    print("Cupy is not installed!")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns
    sns.set_theme()
except:
    pass

np.random.seed(42)

path = r'./data/CoLA/tokenized'


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

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = queries @ keys.swapaxes(-1, -2) / (self.embed_size**(1 / 2))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, axis=-1)

        out = (attention @ values).reshape(N, query_len,
                                           self.heads * self.head_dim)

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm([embed_size])
        self.norm2 = nn.LayerNorm([embed_size])

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


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
        num_classes,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            ) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = Tensor(
            np.repeat(np.arange(0, seq_length).reshape(1, -1), N, 0),
            device=x.device,
        )
        a = self.word_embedding(x)
        b = self.position_embedding(positions)
        out = self.dropout(a + b)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = out.mean(1)
        return self.fc_out(out)


if __name__ == "__main__":
    LR = 1e-5
    EPOCHES = 100
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 512
    use_cuda = True

    device = 'cuda' if pdn.cuda.is_available() and use_cuda else 'cpu'

    X, y = load_data()
    train_X, test_X, train_y, test_y = train_test_split(
        Tensor(X),
        Tensor(y),
        train_size=0.8,
    )

    train_loader = data_loader(
        train_X,
        train_y,
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = data_loader(
        test_X,
        test_y,
        shuffle=False,
        batch_size=TEST_BATCH_SIZE,
    )

    net = Transformer(64, 1, 4, 4, 0.05, X.max(), 44, 2).to(device)
    optimizer = Adam(net.parameters(), lr=LR)
    bar = tqdm(range(EPOCHES))
    info_list = []
    for epoch in bar:

        net.train()

        for batch_X, batch_y in train_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            loss = F.cross_entropy_loss(net(input_, None), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        train_right, train_size = 0, 0
        test_right, test_size = 0, 0

        with pdn.no_grad():
            for batch_X, batch_y in train_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = net(input_, None).argmax(-1)
                train_right += (pred.data == label.data).sum()
                train_size += batch_X.shape[0]

            for batch_X, batch_y in test_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = net(input_, None).argmax(-1)
                test_right += (pred.data == label.data).sum()
                test_size += batch_X.shape[0]

        train_acc, test_acc = train_right / train_size, test_right / test_size
        bar.set_postfix(
            TEST_ACC="{:.4f}".format(test_acc),
            TRAIN_ACC="{:.4f}".format(train_acc),
        )
        info_list.append([train_acc.item(), test_acc.item()])

    print(np.array(info_list))
