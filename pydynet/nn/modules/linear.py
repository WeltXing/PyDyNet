from .module import Module
from ..parameter import Parameter
from .. import init, functional as F
from ...core import Tensor
from ...special import empty
from ...cuda import Device
from ...autograd import no_grad

import math


class Linear(Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {"device": Device(device), "dtype": dtype}
        self.weight = Parameter(
            empty((self.in_features, self.out_features), **kwargs))
        self.bias = Parameter(empty(self.out_features, **
                                    kwargs)) if bias else None
        self.reset_paramters()

    def reset_paramters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return "Linear(in_features={}, out_features={}, bias={})".format(
            self.in_features, self.out_features, self.bias is not None)


class Embedding(Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_embedding = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        kwargs = {"device": Device(device), "dtype": dtype}
        self.weight = Parameter(
            empty((self.num_embedding, self.embedding_dim), **kwargs))

    def forward(self, x: Tensor):
        return F.embedding(x, self.weight, self.padding_idx)

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with no_grad():
                self.weight[self.padding_idx].data = self.weight.xp.zeros(
                    self.weight[self.padding_idx].shape,
                    dtype=self.weight.dtype,
                )
