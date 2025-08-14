import sys

sys.path.append('../pydynet')

import pydynet as pdn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.randn(2)
A = pdn.Tensor([
    [3, 1.],
    [1, 2.],
])
b = pdn.Tensor([-1., 1])


def auto_grad(x, lr: float, n_iter: float):
    Xs, ys = [], []
    x = pdn.Tensor(x, requires_grad=True)

    for _ in range(n_iter):
        obj = x @ A @ x / 2 + b @ x
        obj.backward()

        Xs.append(x.data.copy())
        ys.append(obj.item())
        x.data -= lr * x.grad
        x.zero_grad()

    Xs, ys = np.array(Xs), np.array(ys)
    return Xs[:, 0], Xs[:, 1], ys


def manual_grad(x, lr: float, n_iter: float):
    Xs, ys = [], []

    for _ in range(n_iter):
        obj = x @ A @ x / 2 + b @ x

        Xs.append(x.copy())
        ys.append(obj.item())

        grad = A.data @ x + b.data
        x -= lr * grad

    Xs, ys = np.array(Xs), np.array(ys)
    return Xs[:, 0], Xs[:, 1], ys


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot3D(
    *auto_grad(x, .1, 30),
    color='red',
    lw=0.7,
    label=r'$f(x)=\frac{1}{2}x^\top Ax+b^\top x$',
    marker='^',
    markersize=6,
)

ax1.tick_params(direction='in')
ax1.set_xlim(.45, .60)
ax1.set_ylim(-.8, 0)
ax1.set_zlim(-.8, -.3)
ax1.set_xticks([.45, .5, .55, .6])
ax1.set_yticks([-.8, -.6, -.4, -.2, 0])

plt.title('Gradient descent by AutoGrad')
plt.legend(prop={'size': 11})

ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot3D(
    *manual_grad(x, .1, 30),
    color='blue',
    lw=0.7,
    label=r'$f(x)=\frac{1}{2}x^\top Ax+b^\top x$',
    marker='^',
    markersize=6,
)

ax1.tick_params(direction='in')
ax1.set_xlim(.45, .60)
ax1.set_ylim(-.8, 0)
ax1.set_zlim(-.8, -.3)
ax1.set_xticks([.45, .5, .55, .6])
ax1.set_yticks([-.8, -.6, -.4, -.2, 0])

plt.title('Gradient descent by Manual calculation')
plt.legend(prop={'size': 11})

plt.savefig("imgs/ad2d.png")
