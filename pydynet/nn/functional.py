import numpy as np

from ..core import tensor, function
from .. import unsqueeze, no_grad


def linear(x: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine


def embedding(x: tensor.Tensor, weight: tensor.Tensor, padding_idx: int):
    query = weight[x]
    if padding_idx is not None:
        with tensor.no_grad():
            mask = unsqueeze(x.ne(padding_idx), -1)
        query = query * mask
    return query


def sigmoid(x: tensor.Tensor):
    return tensor.sigmoid(x)


def tanh(x: tensor.Tensor):
    return tensor.tanh(x)


def relu(x: tensor.Tensor):
    return tensor.maximum(0., x)


def leaky_relu(x: tensor.Tensor, alpha: float):
    return tensor.maximum(x, alpha * x)


def silu(x: tensor.Tensor):
    return x / (1 + tensor.exp(-x))


def softmax(x: tensor.Tensor, axis=None):
    '''Softmax函数'''
    with no_grad():
        max_ = x.max(axis, keepdims=True)
    x_sub_max = x - max_
    exp_ = tensor.exp(x_sub_max)
    return exp_ / tensor.sum(exp_, axis=axis, keepdims=True)


def log_softmax(x: tensor.Tensor, axis=None, keepdims=False):
    '''log-softmax函数'''
    with no_grad():
        max_ = x.max(axis, keepdims=True)
    x_sub_max = x - max_
    return x_sub_max - tensor.log(
        tensor.sum(tensor.exp(x_sub_max), axis=axis, keepdims=keepdims))


class __im2col1d(tensor._UnaryOperator):

    def __init__(
        self,
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
    ) -> None:
        self.N, self.in_channels, self.n_features = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_output = (self.n_features - self.kernel_size) // stride + 1
        super().__init__(x)

    def forward_(self, x: tensor.Tensor) -> np.ndarray:
        s0, s1, s2 = x.strides
        shape = (x.shape[0], self.in_channels, self.kernel_size, self.n_output)
        self.__strides = (s0, s1, s2, s2 * self.stride)

        col = self.xp.lib.stride_tricks.as_strided(
            x.data,
            shape=shape,
            strides=self.__strides,
        ).copy()
        return col

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        grad_x = self.xp.zeros(x.shape, dtype=self.dtype)
        view = self.xp.lib.stride_tricks.as_strided(
            grad_x,
            shape=self.shape,
            strides=self.__strides,
        )
        self.xp.add.at(view, (..., ), grad)
        return grad_x


class __pad1d(tensor._UnaryOperator):

    def __init__(self, x: tensor.Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward_(self, x: tensor.Tensor) -> np.ndarray:
        return self.xp.pad(x.data, [(0, 0), (0, 0),
                                    (self.pad_width, self.pad_width)],
                           'constant')

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width]


def conv1d(
    x: tensor.Tensor,
    kernel: tensor.Tensor,
    padding: int = 0,
    stride: int = 1,
):
    '''一维卷积函数

    基于im2col实现的一维卷积.
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_features);
    kernel : Tensor
        卷积核, 形状为(out_channels, in_channels, kernel_size);
    padding : int, default=0
        对输入特征两边补0数量;
    stride : int, default=1
        卷积步长.
    '''
    kernel_size = kernel.shape[-1]
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return (col @ kernel.transpose(1, 2, 0)).sum(1).swapaxes(1, 2)


def max_pool1d(
    x: tensor.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 0,
):
    '''一维池化函数

    基于im2col实现的一维池化.`
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    '''
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.max(-1)


def avg_pool1d(
    x: tensor.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 0,
):
    '''一维平均池化函数

    基于im2col实现的一维池化.`
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    '''
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.mean(-1)


class __im2col2d(tensor._UnaryOperator):

    def __init__(
        self,
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
    ) -> None:
        _, self.in_channels, self.n_h, self.n_w = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_h, self.out_w = (
            self.n_h - self.kernel_size) // self.stride + 1, (
                self.n_w - self.kernel_size) // self.stride + 1

        super().__init__(x)

    def forward_(self, x: tensor.Tensor) -> np.ndarray:
        s0, s1, s2, s3 = x.strides
        shape = (x.shape[0], self.in_channels, self.kernel_size,
                 self.kernel_size, self.out_h, self.out_w)
        self.__strides = (s0, s1, s2, s3, s2 * self.stride, s3 * self.stride)

        col = self.xp.lib.stride_tricks.as_strided(
            x.data,
            shape=shape,
            strides=self.__strides,
        ).copy()
        return col

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        grad_x = self.xp.zeros(x.shape, dtype=self.dtype)
        view = self.xp.lib.stride_tricks.as_strided(
            grad_x,
            shape=self.shape,
            strides=self.__strides,
        )
        self.xp.add.at(view, (..., ), grad)
        return grad_x


class __pad2d(tensor._UnaryOperator):

    def __init__(self, x: tensor.Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward_(self, x: tensor.Tensor) -> np.ndarray:
        return self.xp.pad(x.data, [(0, 0), (0, 0),
                                    (self.pad_width, self.pad_width),
                                    (self.pad_width, self.pad_width)],
                           'constant')

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width,
                    self.pad_width:-self.pad_width]


def conv2d(x: tensor.Tensor,
           kernel: tensor.Tensor,
           padding: int = 0,
           stride: int = 1):
    '''二维卷积函数

    基于im2col实现的二维卷积. 为了实现上的方便, 我们不考虑长宽不同的卷积核, 步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_height, n_width);
    kernel : Tensor
        卷积核, 形状为(out_channels, in_channels, kernel_height, kernel_width);
    padding : int, default=0
        对输入图片周围补0数量;
    stride : int, default=1
        卷积步长.
    '''
    N, _, _, _ = x.shape
    out_channels, _, kernel_size, _ = kernel.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)


def max_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    '''二维卷积函数池化

    基于im2col实现的二维卷积. 为了实现上的方便, 我们不考虑长宽不同的kernel_size, 步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    '''
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.max(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


def avg_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    '''二维平均池化

    基于im2col实现的二维池化. 为了实现上的方便, 我们不考虑长宽不同的kernel_size, 步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据, 形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    '''
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.mean(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


def mse_loss(y_pred, y_true, reduction='mean'):
    '''均方误差'''
    square_sum = function.square(y_pred - y_true)
    if reduction == 'mean':
        return tensor.mean(square_sum)
    elif reduction == 'sum':
        return tensor.sum(square_sum)
    else:
        raise ValueError("reduction must be mean or sum.")


def nll_loss(y_pred, y_true, reduction='mean'):
    '''负对数似然'''
    nll = -y_pred * y_true
    if reduction == 'mean':
        return tensor.mean(nll)
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        raise ValueError("reduction must be mean or sum.")


def cross_entropy_loss(y_pred, y_true, reduction='mean'):
    '''交叉熵损失'''
    update_y_pred = y_pred - y_pred.max().item()
    log_sum_exp = tensor.log(
        tensor.sum(tensor.exp(update_y_pred), 1, keepdims=True))

    neg_log_sm = log_sum_exp - update_y_pred
    if y_true.ndim == 1:
        nll = neg_log_sm[range(len(neg_log_sm)), y_true]
    else:
        nll = neg_log_sm * y_true

    if reduction == 'mean':
        return tensor.mean(nll)
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        raise ValueError("reduction must be mean or sum.")
