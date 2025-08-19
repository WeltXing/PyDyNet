'''优化器类，我们目前实现了\n
- SGD with momentum and Nestrov;\n
- Adagrad;\n
- Adadelta;\n
- Adam.\n

Reference
---------
论文: https://arxiv.org/abs/1609.04747;\n
博客: https://xingcy.net/2021/08/20/gd/.
'''

from math import sqrt
from ..core import Tensor


class Optimizer:
    '''优化器基类'''

    def __init__(self, params: list[Tensor]) -> None:
        self.params: list[Tensor] = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        '''针对self.params梯度清零.'''
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    '''带动量的梯度下降

    Parameters
    ----------
    params : list[Parameter]
        待优化参数;
    lr : float
        学习率;
    momentum : float
        动量系数;
    weight_decay : float, default=0.
        权重衰减系数.
    nesterov : bool, defallt=True.
        是否采用Nesterov加速.
    '''

    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        momentum: float = .5,
        weight_decay: float = 0.,
        nesterov=True,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = [
            param.xp.zeros(param.shape, dtype=param.dtype)
            for param in self.params
        ]

    def step(self):
        for i in range(len(self.params)):
            with self.params[i].device:
                grad = self.params[i].grad + self.weight_decay * self.params[i].data
                self.v[i] *= self.momentum
                self.v[i] += self.lr * grad
                self.params[i].data -= self.v[i]
                if self.nesterov:
                    self.params[i].data -= self.lr * grad


class Adagrad(Optimizer):
    '''Adaptive Gradient Descent
    
    Parameters
    ----------
    params : list[Parameter]
        待优化参数;
    lr : float, default=1e-2.
        学习率;
    weight_decay : float, default=0.
        权重衰减系数.
    eps : float, default=1e-10
        epsilon.
    '''

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        weight_decay: float = 0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.G = [
            param.xp.zeros(param.shape, dtype=param.dtype)
            for param in self.params
        ]

    def step(self):
        for i in range(len(self.params)):
            with self.params[i].device:
                grad = self.params[i].grad + self.weight_decay * self.params[i].data
                self.G[i] += grad**2
                self.params[i].data -= self.lr * grad / (self.eps + self.G[i])**0.5


class Adadelta(Optimizer):
    '''
    Adadelta优化器
    
    params : list[Parameter]
        待优化参数;
    lr : float, default=1e-2.
        学习率;
    rho :float, default=
    weight_decay : float, default=0.
        权重衰减系数.
    eps : float, default=1e-10
        epsilon.
    '''

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.eps = eps
        self.weight_decay = weight_decay
        self.G = [
            param.xp.zeros(param.shape, dtype=param.dtype)
            for param in self.params
        ]

    def step(self):
        for i in range(len(self.params)):
            with self.params[i].device:
                grad = self.params[i].grad + self.weight_decay * self.params[i].data

                self.G[i] = self.rho * self.G[i] + (1 - self.rho) * grad**2
                self.params[i].data -= self.lr * grad / (self.G[i] + self.eps)**0.5


class Adam(Optimizer):

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [
            param.xp.zeros(param.shape, dtype=param.dtype)
            for param in self.params
        ]
        self.v = [
            param.xp.zeros(param.shape, dtype=param.dtype)
            for param in self.params
        ]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            with self.params[i].device:
                grad = self.params[i].grad + self.weight_decay * self.params[i].data
                self.m[i] *= self.beta1
                self.m[i] += (1 - self.beta1) * grad
                self.v[i] *= self.beta2
                self.v[i] += (1 - self.beta2) * grad**2
                a_t = sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
                self.params[i].data -= self.lr * a_t * self.m[i] / (
                    self.v[i]**0.5 + self.eps)
        self.t += 1
