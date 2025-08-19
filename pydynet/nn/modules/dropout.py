from .module import Module
from ...core import Tensor
from ...special import rand


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert p >= 0 and p < 1
        self.p = p

    def forward(self, x) -> Tensor:
        if self._train:
            mask = rand(*x.shape, device=x.device) < 1 - self.p
            return x * mask.astype(x.dtype) / (1 - self.p)
        return x

    def __repr__(self) -> str:
        return "{}(p={})".format(self.__class__.__name__, self.p)
