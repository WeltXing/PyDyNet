import numpy as np
from .core import Tensor


# 一些包装的特殊矩阵
def zeros(shape, dtype=None, device=None, requires_grad=False):
    '''全0张量
    
    Parameters
    ----------
    shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.zeros(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def ones(shape, dtype=None, device=None, requires_grad=False):
    '''全1张量
    
    Parameters
    ----------
    shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.ones(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def randn(*shape, dtype=None, device=None, requires_grad=False):
    '''0-1正态分布张量
    
    Parameters
    ----------
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.randn(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def rand(*shape, dtype=None, device=None, requires_grad=False):
    '''[0, 1)均匀分布张量
    
    Parameters
    ----------
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.rand(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def uniform(low: float,
            high: float,
            shape=None,
            dtype=None,
            device=None,
            requires_grad=False):
    '''均匀分布张量
    
    Parameters
    ----------
    low : float
        均匀分布下界;
    high : float
        均匀分布下界;
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.uniform(low, high, size=shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def empty(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.empty(shape, dtype=dtype),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)
