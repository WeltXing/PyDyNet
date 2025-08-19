from collections import OrderedDict

from ..parameter import Parameter
from ...core import Tensor
from ...autograd import set_grad_enabled
from ...cuda import Device, current_device


class Module:

    def __init__(self) -> None:
        self._train = True
        self.device = Device("cpu")
        self._parameters = OrderedDict()

    def __call__(self, *x) -> Tensor:
        return self.forward(*x)

    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value
        if isinstance(__value, Module):
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]

    def __repr__(self) -> str:
        module_list = [
            module for module in self.__dict__.items()
            if isinstance(module[1], Module)
        ]
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            "\n".join([
                "{:>10} : {}".format(module_name, module)
                for module_name, module in module_list
            ]),
        )

    def parameters(self):
        for param in self._parameters.values():
            if param.requires_grad:
                yield param

    def train(self, mode: bool = True):
        set_grad_enabled(mode)
        self.set_module_state(mode)

    def set_module_state(self, mode: bool):
        self._train = mode
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.set_module_state(mode)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def eval(self):
        return self.train(False)

    def to(self, device):
        if not isinstance(device, Device):
            device = Device(device)
        if self.device != device:
            self.move(device)
        return self

    def move(self, device):
        self.device = device
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.move(device)
            if isinstance(module, Parameter):
                module.to(device)

    def cuda(self):
        return self.to(current_device())

    def cpu(self):
        return self.to('cpu')


class Sequential(Module):

    def __init__(self, *args) -> None:
        super().__init__()
        self.module_list = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.__setattr__(name, module)
                self.module_list.append(module)
        else:
            for idx, module in enumerate(args):
                self.__setattr__(str(idx), module)
                self.module_list.append(module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.module_list:
            x = module(x)
        return x

    def __len__(self):
        return len(self.module_list)


class ModuleList(Module):

    def __init__(self, module_list: list) -> None:
        super().__init__()
        self.module_list = module_list

        for idx, module in enumerate(module_list):
            self.__setattr__(str(idx), module)

    def __getitem__(self, index):
        return self.module_list[index]

    def __len__(self):
        return len(self.module_list)

    def append(self, module):
        self.module_list.append(module)
        self.__setattr__(str(len(self.module_list) - 1), module)

    def index(self, module):
        return self.module_list.index(module)
