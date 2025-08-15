import numpy as np
from .cuda import Device
from .autograd import is_grad_enable, no_grad


class Graph:
    '''计算图, 全局共用一个动态计算图'''
    node_list: list = list()
    size = 0

    @classmethod
    def add_node(cls, node):
        '''添加图节点'''
        cls.node_list.append(node)
        cls.size += 1

    @classmethod
    def free_node(cls, node):
        node.last.clear()

        index = cls.node_list.index(node)
        cls.node_list.pop(index)

        cls.size -= 1


def backward_subroutine(last, node):
    if last.requires_grad:
        add_grad = node.grad_fn(last, node.grad)
        if add_grad.shape != last.shape:
            # handle broadcast
            dim1, dim2 = add_grad.ndim, last.ndim
            axis = (-i for i in range(1, dim2 + 1) if last.shape[-i] == 1)
            add_grad = node.xp.sum(add_grad, axis=tuple(axis), keepdims=True)
            if dim1 != dim2:  # dim1 >= dim2 for sure
                add_grad = add_grad.sum(tuple(range(dim1 - dim2)))
        last.grad += add_grad
    return last


class Tensor:
    '''
    将数据(NumPy数组)包装成可微分张量

    Parameters
    ----------
    data : ndarray
        张量数据, 只要是np.array能够转换的数据;
    dtype : default=None
        数据类型, 和numpy数组的dtype等价
    copy : bool, default=False
        是否拷贝数据
    requires_grad : bool, default=False
        是否需要求梯度;

    Attributes
    ----------
    data : numpy.ndarray
        核心数据, 为NumPy数组;
    requires_grad : bool
        是否需要求梯度;
    grad : numpy.ndarray
        梯度数据, 为和data相同形状的数组(初始化为全0);
    last : list[Tensor]
        上游节点列表.

    Example
    -------
    >>> import numpy as np
    >>> from pydynet.tensor import Tensor
    >>> x = Tensor(1., requires_grad=True)
    >>> y = Tensor([1, 2, 3], dtype=float)
    >>> z = Tensor(np.random.rand(3, 4))
    '''

    def __init__(
        self,
        data,
        dtype=None,
        copy: bool = True,
        device: Device | int | str | None = None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            assert 0

        self.copy = copy
        self.device = Device(device)
        with self.device:
            self.data = self.xp.array(data, dtype=dtype, copy=self.copy)
            # self.data = self.xp.asarray(data, dtype=dtype)

        self.requires_grad = is_grad_enable() and requires_grad
        if self.requires_grad:
            if not self.xp.issubdtype(self.dtype, np.floating):
                raise TypeError(
                    "Only Tensors of floating point dtype can require gradients!"
                )
            self.grad = self.xp.zeros(self.shape, dtype=dtype)
            self.last: list[Tensor] = list()
            Graph.add_node(self)
        else:
            self.grad = None

    @property
    def is_leaf(self) -> bool:
        '''判断是否为叶节点:需要求导且无上游节点的节点为叶节点.'''
        return not self.requires_grad or len(self.last) == 0

    @property
    def shape(self) -> tuple[int]:
        '''张量的形状, 用法同NumPy.
        
        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).shape
        (1, 2)
        '''
        return self.data.shape

    @property
    def ndim(self) -> int:
        '''张量的维度, 用法同NumPy.
        
        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).ndim
        2
        '''
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        '''张量的数据类型, 用法同NumPy.

        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).dtype
        dtype('int64')
        '''
        return self.data.dtype

    @property
    def size(self) -> int:
        '''张量的元素个数, 用法同NumPy.

        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[1, 1]]).size
        2
        '''
        return self.data.size

    @property
    def strides(self) -> tuple[int]:
        return self.data.strides

    @property
    def T(self):
        return self.transpose()

    def astype(self, new_type):
        '''类型转换, 我们不允许可求导节点的类型转换'''
        with self.device:
            self.data = self.data.astype(new_type)
            self.dtype = self.data.dtype
        return self

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def swapaxes(self, axis1: int, axis2: int):
        return swapaxes(self, axis1, axis2)

    def vsplit(self, indices_or_sections: int | tuple):
        return vsplit(self, indices_or_sections)

    def hsplit(self, indices_or_sections: int | tuple):
        return hsplit(self, indices_or_sections)

    def dsplit(self, indices_or_sections: int | tuple):
        return dsplit(self, indices_or_sections)

    def split(self, indices_or_sections: int | tuple, axis=0):
        return split(self, indices_or_sections, axis=axis)

    def max(self, axis: int | tuple | None = None, keepdims: bool = False):
        return max(self, axis, keepdims)

    def min(self, axis: int | tuple | None = None, keepdims: bool = False):
        return min(self, axis, keepdims)

    def mean(self, axis: int | tuple | None = None, keepdims: bool = False):
        return mean(self, axis, keepdims)

    def sum(self, axis: int | tuple | None = None, keepdims: bool = False):
        return sum(self, axis, keepdims)

    def argmax(self, axis: int | tuple | None = None, keepdims: bool = False):
        return argmax(self, axis, keepdims)

    def argmin(self, axis: int | tuple | None = None, keepdims: bool = False):
        return argmin(self, axis, keepdims)

    def build_edge(self, node):
        '''构建节点的有向边, 正常不适用'''
        node.last.append(self)

    def __repr__(self) -> str:
        return "{}({}, requires_grad={}".format(
            "Tensor",
            self.data,
            self.requires_grad,
        ) + (", device={}".format(self.device)
             if self.device.device != "cpu" else "") + ")"

    def __add__(self, x):
        return add(self, x)

    def __radd__(self, x):
        return add(x, self)

    def __sub__(self, x):
        return sub(self, x)

    def __rsub__(self, x):
        return sub(x, self)

    def __mul__(self, x):
        return mul(self, x)

    def __rmul__(self, x):
        return mul(x, self)

    def __matmul__(self, x):
        return matmul(self, x)

    def __rmatmul__(self, x):
        return matmul(x, self)

    def __truediv__(self, x):
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def __setitem__(self, key, value):
        '''
        重载了切片/索引赋值的操作, 我们不允许self允许求导, 否则将出现错误

        Parameters
        ----------
        key : 索引, 支持NumPy的数字、切片和条件索引
        value : 值, 可以是NumPy数字, 也可以是数字

        Example
        -------
        >>> x = Tensor([1, 2, 3])
        >>> x[x <= 2] = 0
        >>> x
        <[0 0 3], int64, Tensor>
        '''
        if is_grad_enable():
            raise ValueError(
                "In-place operation is forbidden in node requires grad.")
        if isinstance(key, Tensor):
            key = key.data

        with self.device:
            self.data[key] = value.data if isinstance(value, Tensor) else value

    def __len__(self) -> int:
        return len(self.data)

    def __inplace(self, other, func):
        if self.requires_grad:
            raise ValueError(
                "In-place operation is forbidden in node requires grad.")
        if isinstance(other, Tensor):
            other = other.data
        with self.device:
            self.data[...] = func(self.data, other)
        return self

    def __iadd__(self, other):
        return self.__inplace(other, lambda x, y: x + y)

    def __isub__(self, other):
        return self.__inplace(other, lambda x, y: x - y)

    def __imul__(self, other):
        return self.__inplace(other, lambda x, y: x * y)

    def __itruediv__(self, other):
        return self.__inplace(other, lambda x, y: x / y)

    def __imatmul__(self, other):
        return self.__inplace(other, lambda x, y: x @ y)

    def __compare(self, other, func):
        if isinstance(other, Tensor):
            other = other.data

        with self.device:
            return Tensor(func(self.data, other), self.xp.bool_, self.device)

    @no_grad()
    def __lt__(self, other):
        return self.__compare(other, lambda x, y: x < y)

    @no_grad()
    def __le__(self, other):
        return self.__compare(other, lambda x, y: x <= y)

    @no_grad()
    def __gt__(self, other):
        return not self.__le__(other)

    @no_grad()
    def __ge__(self, other):
        return not self.__lt__(other)

    def backward(self, retain_graph: bool = False):
        '''
        以节点为输出进行反向传播

        Parameters
        ----------
        retain_graph : bool, default=False
            是否保留计算图

        Example
        -------
        >>> from pydynet.tensor import Tensor
        >>> import pydynet.functional as F
        >>> x = Tensor(2., requires_grad=True)
        >>> y = x**2 + x - 1
        >>> y.backward()
        >>> x.grad
        5.
        '''
        if self not in Graph.node_list:
            raise ValueError(
                "Auto-grad is failed because current node is not in graph.")

        if self.size > 1:
            raise ValueError("backward should be called only on a scalar.")

        self.grad = self.xp.ones(self.shape, dtype=self.dtype)
        y_id = Graph.size - Graph.node_list[::-1].index(self) - 1
        for node in Graph.node_list[y_id::-1]:
            for last in node.last:
                backward_subroutine(last, node)

            # if not retain graph and node is not leaf, free it
            if not retain_graph and not node.is_leaf:
                Graph.free_node(node)

    def zero_grad(self):
        '''梯度归零'''
        self.grad[...] = 0

    def numpy(self) -> np.ndarray:
        '''返回Tensor的内部数据, 即NumPy数组(拷贝)'''
        data = self.data
        if self.device != 'cpu':
            data = data.get()
        return data.copy()

    def item(self):
        return self.data.item()

    def to(self, device):
        device = Device(device)
        if self.device != device:
            self.device = device
            with device:
                self.data = self.xp.asarray(self.data)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda:0")

    @property
    def xp(self):
        return self.device.xp


class UnaryOperator(Tensor):
    '''
    一元运算算子的基类, 将一个一元函数抽象成类

    Example
    -------
    >>> class exp(UnaryOperator):
            def forward(self, x: Tensor):
                return np.exp(x.data)
            def grad_fn(self, x: Tensor, grad) -> np.ndarray:
                return self.data * grad
    >>> x = Tensor(1.)
    >>> y = exp(x)
    '''

    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.device = x.device
        data = self.forward(x)
        super().__init__(
            data=data,
            device=self.device,
            copy=None,
            dtype=data.dtype,
            requires_grad=is_grad_enable() and x.requires_grad,
        )
        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        with self.device:
            return self.forward_(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        '''前向传播函数, 参数为Tensor, 返回的是NumPy数组'''
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        '''
        反向传播函数, 参数为下游节点, 从上游流入该节点梯度。
        注："上游"和"下游"针对的是反向传播, 比如z = f(x, y), x和y是z的下游节点.

        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class BinaryOperator(Tensor):
    '''
    二元运算算子的基类, 将一个二元函数抽象成类

    Example
    -------
    >>> add(BinaryOperator):
            def forward(self, x: Tensor, y: Tensor):
                return x.data + y.data
            def grad_fn(self, node: Tensor, grad: np.ndarray):
                return np.ones(self.shape) * grad
    >>> x = Tensor(1.)
    >>> y = Tensor(2.)
    >>> z = add(x, y)
    '''

    def __init__(self, x: Tensor, y: Tensor) -> None:
        # A strict type-cast: follow the Tensor
        if not isinstance(x, Tensor) and isinstance(y, Tensor):
            x = Tensor(x, dtype=y.dtype, device=y.device)
        elif isinstance(x, Tensor) and not isinstance(y, Tensor):
            y = Tensor(y, dtype=x.dtype, device=x.device)
        elif not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            x, y = Tensor(x), Tensor(y)
        assert x.device == y.device
        # if x.device != 'cpu':
        #     assert x.data.device.id == y.data.device.id
        self.device = x.device
        data = self.forward(x, y)

        super().__init__(
            data=data,
            copy=None,
            device=x.device,
            dtype=data.dtype,
            requires_grad=is_grad_enable()
            and (x.requires_grad or y.requires_grad),
        )
        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        with self.device:
            return self.forward_(x, y)

    def forward_(self, x: Tensor, y: Tensor) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        '''
        反向传播函数, 参数为下游节点, 从上游流入该节点梯度。
        注："上游"和"下游"针对的是反向传播, 比如z = f(x, y), x和y是z的下游节点.

        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class add(BinaryOperator):
    '''
    加法算子

    Example
    -------
    >>> x = Tensor(1.)
    >>> y = Tensor(2.)
    >>> z = add(x, y) # 在Tensor类中进行了重载, 所以也可以写成
    >>> z = x + y
    '''

    def forward_(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return grad[...]


class sub(BinaryOperator):
    '''
    减法算子, 在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''

    def forward_(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return grad[...]
        return -grad


class mul(BinaryOperator):
    '''
    元素级乘法算子, 在Tensor类中进行重载

    Example
    -------
    >>> x = Tensor([1., 2.])
    >>> y = Tensor([2., 3.])
    >>> z = mul(x, y) # [2, 6]

    See also
    --------
    add : 加法算子
    '''

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward_(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return self.last[1].data * grad
        return self.last[0].data * grad


class div(BinaryOperator):
    '''
    除法算子, 在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward_(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.last[1].data
        if node is self.last[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    '''
    幂运算算子, 在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward_(self, x: Tensor, y: Tensor):
        return x.data**y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * self.xp.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    '''
    矩阵乘法算子, 在Tensor类中进行重载, 张量的矩阵乘法遵从NumPy Matmul的规则.

    参考 : https://xingcy.net/2022/04/26/broadcast/

    See also
    --------
    add : 加法算子
    '''

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward_(self, x: Tensor, y: Tensor) -> np.ndarray:
        self.expand_a, self.expand_b = x.ndim < 2, y.ndim < 2
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        # regularization for input shape
        if self.expand_a:
            grad = self.xp.expand_dims(grad, 0)

        if self.expand_b:
            grad = self.xp.expand_dims(grad, -1)

        if node is self.last[0]:
            grad1 = grad @ (self.xp.atleast_2d(self.last[1].data) if self.
                            expand_b else self.last[1].data.swapaxes(-1, -2))
            if self.expand_a:
                grad1 = grad1[0]
            return grad1
        else:
            grad2 = (self.xp.atleast_2d(self.last[0].data) if self.expand_a
                     else self.last[0].data).swapaxes(-1, -2) @ grad
            if self.expand_b:
                grad2 = grad2[..., 0]
            return grad2


class abs(UnaryOperator):
    '''
    绝对值算子, 在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''

    def forward_(self, x: Tensor) -> np.ndarray:
        return self.xp.abs(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        mask = self.xp.zeros(x.shape, dtype=x.dtype)
        mask[x.data > 0] = 1.
        mask[x.data < 0] = -1.
        return grad * mask


class sum(UnaryOperator):
    '''
    求和算子, 在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求和方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    Example
    -------
    >>> x = Tensor(
            [[1, 2, 3],
            [4, 5, 6]]
        )
    >>> s1 = x.sum(0) # [5, 7, 9]
    >>> s2 = x.sum(1) # [6, 15]
    >>> s3 = sum(x, keepdims=True) # [[21]]
    ```
    '''

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return self.xp.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(x.shape, dtype=x.dtype) * grad


class mean(UnaryOperator):
    '''
    求均值算子, 在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求均值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    sum : 求和算子
    '''

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return self.xp.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(
            x.shape, dtype=x.dtype) * grad * self.data.size / x.data.size


class max(UnaryOperator):
    '''
    求最大值算子, 在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求最大值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    sum : 求和算子
    '''

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return self.xp.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(x.dtype) * grad


class min(UnaryOperator):
    '''
    求最小值算子, 在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求最大值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    max : 求最大值算子
    '''

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return self.xp.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(x.dtype) * grad


class argmax(Tensor):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.keepdims = keepdims
        self.device = x.device
        super().__init__(self.forward(x), copy=False, device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        with self.device:
            return self.xp.argmax(
                x.data,
                axis=self.axis,
                keepdims=self.keepdims,
            )


class argmin(Tensor):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.keepdims = keepdims
        self.device = x.device
        super().__init__(self.forward(x), copy=False, device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        with self.device:
            return self.xp.argmin(
                x.data,
                axis=self.axis,
                keepdims=self.keepdims,
            )


class exp(UnaryOperator):
    '''指数运算
    
    Example
    -------
    >>> x = Tensor(1.)
    >>> y = exp(x)
    '''

    def forward_(self, x: Tensor):
        return self.xp.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    '''对数运算
    
    Example
    -------
    >>> x = Tensor(1.)
    >>> y = log(x)
    '''

    def forward_(self, x: Tensor):
        return self.xp.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):

    def forward_(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):

    def forward_(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


# 非计算函数
class reshape(UnaryOperator):
    '''
    张量形状变换算子, 在Tensor中进行重载

    Parameters
    ----------
    new_shape : tuple
        变换后的形状, 用法同NumPy
    '''

    def __init__(self, x: Tensor, new_shape: tuple) -> None:
        self.new_shape = new_shape
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
    '''
    张量转置算子, 在Tensor中进行重载(Tensor.T和Tensor.transpose)

    Parameters
    ----------
    axes : tuple
        转置的轴变换, 用法同NumPy
    '''

    def __init__(self, x: Tensor, axes: tuple = None) -> None:
        self.axes = axes
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return x.data.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class swapaxes(UnaryOperator):
    '''
    张量交换轴算子

    Parameters
    ----------
    axis1 : int
        第一个axis;
    axis2 : int
        第二个axis.
    '''

    def __init__(self, x: Tensor, axis1: int, axis2: int) -> None:
        self.axis1 = axis1
        self.axis2 = axis2
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return x.data.swapaxes(self.axis1, self.axis2)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.swapaxes(self.axis1, self.axis2)


class get_slice(UnaryOperator):
    '''
    切片算子, 为Tensor类提供索引和切片接口

    Example
    -------
    >>> x = Tensor(
            np.arange(12).reshape(3, 4).astype(float),
            requires_grad=True,
        )
    >>> y = x[:2, :2].sum()
    >>> y.backward()
    >>> x.grad 
    [[1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [0. 0. 0. 0.]]
    '''

    def __init__(self, x: Tensor, key) -> None:
        if isinstance(key, tuple):
            new_key = []
            for k in key:
                new_key.append(k if not isinstance(k, Tensor) else k.data)
            self.key = tuple(new_key)
        elif isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward_(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        full_grad = self.xp.zeros(x.shape, dtype=x.dtype)
        full_grad[self.key] = grad
        return full_grad


class concat(Tensor):
    '''对多个张量进行连接, 用法类似于`numpy.concatenate`
    
    Parameters
    ----------
    tensors : 
        待连接的张量：
    axis : default=0
        连接轴, 默认是沿着第一个轴拼接.
    '''

    def __init__(self, tensors: list[Tensor], axis=0) -> None:
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(
                tensors[i],
                Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"
            if i == 0:
                device = tensors[i].device
            else:
                assert tensors[i].device == device
            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])
        self.device = device
        super().__init__(
            self.forward(),
            requires_grad=requires_grad and is_grad_enable(),
            copy=False,
            device=self.device,
        )
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        with self.device:
            return self.xp.concatenate([t.data for t in self.tensors],
                                       axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


def sqrt(x: Tensor):
    '''平方根函数'''
    return x**0.5


def square(x: Tensor):
    '''平方函数'''
    return x * x


def vsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[0]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[0]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[st:end])

    return sub_tensors


def hsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[1]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[1]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, st:end])

    return sub_tensors


def dsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[2]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[2]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, :, st:end])

    return sub_tensors


def split(
    x: Tensor,
    indices_or_sections: int | tuple,
    axis: int = 0,
) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    if axis == 0 or axis == -x.ndim:
        return vsplit(x, indices_or_sections)
    elif axis == 1 or axis == -x.ndim + 1:
        return hsplit(x, indices_or_sections)
    elif axis == 2 or axis == -x.ndim + 2:
        return dsplit(x, indices_or_sections)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[axis]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[axis]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    stensor = swapaxes(x, 0, axis)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(swapaxes(stensor[st:end], axis, 0))
    return sub_tensors


def unsqueeze(x: Tensor, axis):
    '''等价于numpy的expand_dims, 因此我们借用了expand_dims的源码'''
    from numpy.core.numeric import normalize_axis_tuple

    if type(axis) not in (tuple, list):
        axis = (axis, )

    out_ndim = len(axis) + x.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(x.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return x.reshape(*shape)
