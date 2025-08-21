from .module import Module
from ..parameter import Parameter
from .. import init
from .. import functional as F
from ...special import empty
from ...cuda import Device

import math


class Conv1d(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": Device(device), "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = Parameter(
            empty((self.out_channels, self.in_channels, self.kernel_size),
                  **kwargs))
        self.bias = Parameter(empty(
            (1, self.out_channels, 1), **kwargs)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv1d = F.conv1d(x, self.weight, self.padding, self.stride)
        if self.bias is not None:
            return conv1d + self.bias
        return conv1d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )


class Conv2d(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": Device(device), "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = Parameter(
            empty((self.out_channels, self.in_channels, self.kernel_size,
                   self.kernel_size), **kwargs))
        self.bias = Parameter(empty(
            (1, self.out_channels, 1, 1), **kwargs)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv2d = F.conv2d(x, self.weight, self.padding, self.stride)
        if self.bias is not None:
            return conv2d + self.bias
        return conv2d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )
