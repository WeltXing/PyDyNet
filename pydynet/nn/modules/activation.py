from .module import Module
from .. import functional as F
from ...core import Tensor


class Sigmoid(Module):
    '''激活函数层 : Sigmoid'''

    def forward(self, x) -> Tensor:
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Tanh(Module):
    '''激活函数层 : Tanh'''

    def forward(self, x) -> Tensor:
        return F.tanh(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class ReLU(Module):
    '''激活函数层 : ReLU'''

    def forward(self, x) -> Tensor:
        return F.relu(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class LeakyReLU(Module):
    '''
    激活函数层 : LeakyReLU
    
    Parameter
    ---------
    alpha : float
        负输入对应的斜率.
    '''

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x) -> Tensor:
        return F.leaky_relu(x, self.alpha)

    def __repr__(self) -> str:
        return "{}(alpha={})".format(self.__class__.__name__, self.alpha)


class Softmax(Module):
    '''
    激活函数层 : softmax

    Parameter
    ---------
    axis : Optional[Tuple[int]], default=None
        沿着axis计算softmax.
    '''

    def __init__(self, axis=None) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x) -> Tensor:
        return F.softmax(x, self.axis)

    def __repr__(self) -> str:
        return "{}(axis={})".format(self.__class__.__name__, self.axis)
