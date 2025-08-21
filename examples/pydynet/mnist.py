import gzip, argparse
from os.path import join
from tqdm import tqdm

import numpy as np
import pydynet as pdn
from pydynet import nn
import pydynet.nn.functional as F
from pydynet.optim import Adam
from pydynet.data import data_loader


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
            data = np.frombuffer(buffer, dtype=np.uint8)
            # Normalize the data to be in the range [0, 1]
            data = data / 255.0
            # Reshape the data to be in the shape (number_of_images, 28, 28)
            data = data.reshape(-1, 1, 28, 28)
            return pdn.Tensor(data).astype(DTYPE)

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

    def forward(self, x):  # for batch only
        return x.reshape(x.shape[0], -1)


class MLP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28, 1024, dtype=DTYPE),
        )
        self.layer2 = nn.Linear(1024, 1024, dtype=DTYPE)
        self.layer3 = nn.Linear(1024, 10, dtype=DTYPE)

    def forward(self, x):
        z1 = F.relu(self.layer1(x))
        z2 = F.relu(self.layer2(z1))
        return self.layer3(z2)


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1, 1, dtype=DTYPE)
        self.conv2 = nn.Conv2d(20, 50, 3, 1, 1, dtype=DTYPE)
        self.fc1 = nn.Linear(7 * 7 * 50, 500, dtype=DTYPE)
        self.fc2 = nn.Linear(500, 10, dtype=DTYPE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 7 * 7 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


parser = argparse.ArgumentParser()
parser.add_argument("--network",
                    help="Network structure",
                    choices=['mlp', 'conv'],
                    default='conv')
parser.add_argument('--batch-size',
                    type=int,
                    default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=1024,
                    metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help='random seed (default: 1)')
args = parser.parse_args()

DTYPE = np.float32
np.random.seed(args.seed)
device = f'cuda:{pdn.cuda.device_count() - 1}' if pdn.cuda.is_available(
) and not args.no_cuda else 'cpu'

net = {'mlp': MLP(), 'conv': ConvNet()}.get(args.network).to(device)
print(net)

optimizer = Adam(net.parameters(), lr=args.lr)

dataset = MNISTDataset(r'./examples/data/MNIST/raw')
train_loader = data_loader(
    *dataset.load_train(),
    shuffle=True,
    batch_size=args.batch_size,
)
test_loader = data_loader(
    *dataset.load_test(),
    shuffle=False,
    batch_size=args.test_batch_size,
)

bar = tqdm(range(args.epochs))
info_list = []
for epoch in bar:

    net.train()

    for batch_X, batch_y in train_loader:
        input_, label = batch_X.to(device), batch_y.to(device)
        loss = F.cross_entropy_loss(net(input_), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()

    train_right, train_size = 0, 0
    test_right, test_size = 0, 0
    with pdn.no_grad():
        for batch_X, batch_y in train_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            pred: pdn.Tensor = net(input_).argmax(-1)
            train_right += pred.eq(label).sum().item()
            train_size += batch_X.shape[0]

        for batch_X, batch_y in test_loader:
            input_, label = batch_X.to(device), batch_y.to(device)
            pred = net(input_).argmax(-1)
            test_right += pred.eq(label).sum().item()
            test_size += batch_X.shape[0]

    train_acc, test_acc = train_right / train_size, test_right / test_size
    bar.set_postfix(TEST_ACC="{:.4f}".format(test_acc),
                    TRAIN_ACC="{:.4f}".format(train_acc),
                    LOSS="{:.6f}".format(loss.item()))
