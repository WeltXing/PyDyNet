import sys

sys.path.append('../pydynet')

import numpy as np
try:
    import cupy as cp
except:
    print("Cupy is not installed!")

import gzip
from os.path import join
from tqdm import tqdm

import pydynet as pdn
import pydynet.nn as nn
import pydynet.nn.functional as F
from pydynet.optim import Adam
from pydynet.data import data_loader

from warnings import filterwarnings

filterwarnings('ignore')


class MNISTDataset:

    def __init__(self, root) -> None:
        self.root = root
        self.train_images_path = join(root, 'train-images-idx3-ubyte.gz')
        self.train_labels_path = join(root, 'train-labels-idx1-ubyte.gz')
        self.test_images_path = join(root, 't10k-images-idx3-ubyte.gz')
        self.test_labels_path = join(root, 't10k-labels-idx1-ubyte.gz')

    def load_train(self):
        return (
            MNISTDataset.load_mnist_images(self.train_images_path),
            MNISTDataset.load_mnist_labels(self.train_labels_path),
        )

    def load_test(self):
        return (
            MNISTDataset.load_mnist_images(self.test_images_path),
            MNISTDataset.load_mnist_labels(self.test_labels_path),
        )

    @staticmethod
    def load_mnist_images(file_path):
        with gzip.open(file_path, 'r') as f:
            # Skip the magic number and dimensions (4 bytes magic number + 4 bytes each for dimensions)
            f.read(16)
            # Read the rest of the file
            buffer = f.read()
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            # Normalize the data to be in the range [0, 1]
            data = data / 255.0
            # Reshape the data to be in the shape (number_of_images, 28, 28)
            data = data.reshape(-1, 1, 28, 28)
            return pdn.Tensor(data)

    @staticmethod
    def load_mnist_labels(file_path):
        with gzip.open(file_path, 'r') as f:
            # Skip the magic number and number of items (4 bytes magic number + 4 bytes number of items)
            f.read(8)
            # Read the rest of the file
            buffer = f.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return pdn.Tensor(labels, dtype=int)


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):  # for batch only
        return x.reshape(x.shape[0], -1)


class ResidualMLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28, 1024),
        )
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 10)

    def forward(self, x):
        z1 = F.relu(self.layer1(x))
        z2 = F.relu(self.layer2(z1))
        return self.layer3(z1 + z2)


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 7 * 7 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    LR = 5e-3
    EPOCHES = 50
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 512
    use_cuda = True

    device = 'cuda' if pdn.cuda.is_available() and use_cuda else 'cpu'

    # net = ResidualMLP().to(device)
    net = ConvNet().to(device)
    print(net)

    optimizer = Adam(net.parameters(), lr=LR)

    dataset = MNISTDataset(r'./data/MNIST/raw')
    train_loader = data_loader(
        *dataset.load_train(),
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = data_loader(
        *dataset.load_test(),
        shuffle=False,
        batch_size=TEST_BATCH_SIZE,
    )

    bar = tqdm(range(EPOCHES))
    info_list = []
    for epoch in bar:

        net.train()

        for batch_X, batch_y in train_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            loss = F.cross_entropy_loss(net(input_), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation, no grad
        net.eval()

        train_right, train_size = 0, 0
        test_right, test_size = 0, 0
        with pdn.no_grad():
            for batch_X, batch_y in train_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = net(input_).argmax(-1)
                train_right += (pred.data == label.data).sum()
                train_size += batch_X.shape[0]

            for batch_X, batch_y in test_loader:
                input_, label = batch_X.to(device), batch_y.to(device)
                pred = net(input_).argmax(-1)
                test_right += (pred.data == label.data).sum()
                test_size += batch_X.shape[0]

        train_acc, test_acc = train_right / train_size, test_right / test_size
        bar.set_postfix(
            TEST_ACC="{:.4f}".format(test_acc),
            TRAIN_ACC="{:.4f}".format(train_acc),
        )
        info_list.append([train_acc.item(), test_acc.item()])

    print(np.array(info_list))
