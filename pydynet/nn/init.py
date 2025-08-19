from ..core import Tensor
from ..autograd import no_grad
import math


def calculate_gain(nonlinearity: str, param: float = None) -> float:
    return {
        "linear": 1,
        "conv1d": 1,
        "conv2d": 1,
        "sigmoid": 1,
        "tanh": 5 / 3,
        "relu": math.sqrt(2.),
        "leaky_relu":
        math.sqrt(2. / (1 + (param if param != None else 0.01)**2))
    }[nonlinearity]


def _calculate_fan(tensor: Tensor):
    assert tensor.ndim >= 2
    fan_in, fan_out = tensor.shape[:2]
    if tensor.ndim > 2:
        receptive_field_size = math.prod(tensor.shape[2:])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    return fan_in, fan_out


@no_grad()
def uniform_(tensor: Tensor, a=0., b=1.) -> Tensor:
    tensor.data[...] = tensor.xp.random.uniform(a, b, tensor.shape)
    return tensor


@no_grad()
def normal_(tensor: Tensor, mean=0., std=1.) -> Tensor:
    tensor.data[...] = tensor.xp.random.normal(mean, std, size=tensor.shape)
    return tensor


@no_grad()
def constant_(tensor: Tensor, val: float) -> Tensor:
    tensor.data[...] = val
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.)


def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.)


def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    bound = gain * math.sqrt(6. / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return normal_(tensor, std=std)


def kaiming_uniform_(tensor: Tensor,
                     a: float = 0.,
                     mode='fan_in',
                     nonlinearity='relu') -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = {
        "fan_in": fan_in,
        "fan_out": fan_out,
    }[mode]
    gain = calculate_gain(nonlinearity, a)
    bound = gain * math.sqrt(3. / fan)
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor,
                    a: float = 0.,
                    mode='fan_in',
                    nonlinearity='relu'):
    fan_in, fan_out = _calculate_fan(tensor)
    fan = {
        "fan_in": fan_in,
        "fan_out": fan_out,
    }[mode]
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, std=std)
