import sys

sys.path.append('../pydynet')

import pydynet as pdn
import numpy as np
import matplotlib.pyplot as plt


def auto_grad(x: float, lr: float, n_iter: int):
    x_list = [x]
    x = pdn.Tensor(float(x), requires_grad=True)

    for _ in range(n_iter):
        x.zero_grad()
        y = pdn.log((x - 7)**2 + 6)
        y.backward()

        x.data -= lr * x.grad
        x_list.append(x.item())

    return x_list


def manual_grad(x: float, lr: float, n_iter: int):
    x_list = [x]
    for _ in range(n_iter):
        grad = 2 * (x - 7) / ((x - 7)**2 + 6)
        x -= lr * grad

        x_list.append(x)

    return x_list

if __name__ == "__main__":
    x_ = np.linspace(0, 10, 101)
    f = np.log((x_ - 7)**2 + 6)

    x1 = np.array(auto_grad(1., 1.5, 20))
    x2 = np.array(manual_grad(1., 1.5, 20))
    y1 = np.log((x1 - 7)**2 + 6)
    y2 = np.log((x2 - 7)**2 + 6)

    plt.figure(figsize=(9, 3))

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 0.5

    plt.subplot(1, 2, 1)
    plt.grid()

    plt.xlim(0, 10)
    plt.ylim(1.5, 4)
    plt.plot(x_, f, label=r"$f(x)=\log((x-7)^2+10)$", color='blue', lw=.7)
    plt.scatter(x1,
                y1,
                color='red',
                marker='^',
                s=50,
                zorder=10,
                label='Gradient descent with lr=1.5')

    plt.yticks([1.5, 2, 2.5, 3, 3.5, 4], size=13)
    plt.xticks([2, 4, 6, 8, 10], size=13)
    plt.title("Gradient descent by AutoGrad")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.grid()

    plt.xlim(0, 10)
    plt.ylim(1.5, 4)
    plt.plot(x_, f, label=r"$f(x)=\log((x-7)^2+10)$", color='blue', lw=.7)
    plt.scatter(x1,
                y1,
                color='green',
                marker='*',
                s=50,
                zorder=10,
                label='Gradient descent with lr=1.5')
    plt.yticks([1.5, 2, 2.5, 3, 3.5, 4], size=13)
    plt.xticks([2, 4, 6, 8, 10], size=13)
    plt.title("Gradient descent by Manual calculation")
    plt.legend()

    plt.savefig("imgs/ad1d.png")
