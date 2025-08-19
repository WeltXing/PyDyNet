from ..core import Tensor


class Parameter(Tensor):

    def __init__(self, data: Tensor, requires_grad: bool = True) -> None:
        super().__init__(
            data=data.data,
            dtype=data.dtype,
            device=data.device,
            copy=False,
            requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        return "Parameter : \n{}".format(self.data) + (",\ndevice={}".format(
            self.device) if self.device.device != "cpu" else "")
