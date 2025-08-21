import sys, pytest, random
import numpy as np
from itertools import product

sys.path.append('../pydynet')

import pydynet as pdn

np.random.seed(0)
random.seed(0)

type_list = [np.float16, np.float32, np.float64]


def matmul_shape_pair(max_dim=4, max_size=5):
    ndim = random.randint(0, max_dim)

    shape1 = []
    shape2 = []
    for _ in range(ndim):
        if random.random() < 0.5:
            # 50% 概率设置为 1, 确保广播可能
            s1, s2 = random.choice([(1, random.randint(1, max_size)),
                                    (random.randint(1, max_size), 1)])
        else:
            # 否则两边相同
            val = random.randint(1, max_size)
            s1, s2 = val, val
        shape1.append(s1)
        shape2.append(s2)
    shape1, shape2 = tuple(shape1), tuple(shape2)

    m = random.randint(1, max_size)
    n = random.randint(1, max_size)
    p = random.randint(1, max_size)

    shape1 = shape1 + (m, n)
    shape2 = shape2 + (n, p)

    shape1 = shape1[random.randint(0, len(shape1) - 2):]

    return shape1, shape2


def broadcastable_shape_pair(max_dim=4, max_size=5):
    ndim = random.randint(0, max_dim)  # 随机维数
    shape1 = []
    shape2 = []
    for _ in range(ndim):
        if random.random() < 0.5:
            # 50% 概率设置为 1, 确保广播可能
            s1, s2 = random.choice([(1, random.randint(1, max_size)),
                                    (random.randint(1, max_size), 1)])
        else:
            # 否则两边相同
            val = random.randint(1, max_size)
            s1, s2 = val, val
        shape1.append(s1)
        shape2.append(s2)
    shape1, shape2 = tuple(shape1), tuple(shape2)

    # 随机缺失维度
    shape1 = shape1[random.randint(0, len(shape1)):]
    return shape1, shape2


def array_pair_generator(pair_gen_func,
                         max_dim=4,
                         max_size=5,
                         n_iter=4,
                         seed=None):
    rng = np.random.default_rng(seed)
    count = 0
    while n_iter is None or count < n_iter:
        shape1, shape2 = pair_gen_func(max_dim, max_size)
        a = rng.standard_normal(size=shape1).astype(rng.choice(type_list))
        b = rng.standard_normal(size=shape2).astype(rng.choice(type_list))
        yield a, b
        count += 1


test_list = array_pair_generator(broadcastable_shape_pair, 4, 5, 8, seed=42)
func_list = [(pdn.add, np.add), (pdn.sub, np.subtract), (pdn.mul, np.multiply),
             (pdn.div, np.divide), (pdn.pow, np.power),
             (pdn.maximum, np.maximum), (pdn.minimum, np.minimum)]
test_list = [(*array, *funcs)
             for (array, funcs) in product(test_list, func_list)]


@pytest.mark.parametrize("operand1, operand2, pdn_func, np_func", test_list)
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_binary_operator(operand1: np.ndarray, operand2: np.ndarray,
                         pdn_func: callable, np_func: callable):
    pdn_operand1, pdn_operand2 = pdn.Tensor(operand1), pdn.Tensor(operand2)
    pdn_output: pdn.Tensor = pdn_func(pdn_operand1, pdn_operand2)
    np_output: np.ndarray = np_func(operand1, operand2)
    assert pdn_output.shape == np_output.shape
    assert pdn_output.dtype == np_output.dtype
    assert np.allclose(pdn_output.data, np_output, equal_nan=True)


test_list = array_pair_generator(matmul_shape_pair, 4, 5, 8, seed=42)


@pytest.mark.parametrize("operand1, operand2", test_list)
def test_matmul(operand1: np.ndarray, operand2: np.ndarray):
    pdn_operand1, pdn_operand2 = pdn.Tensor(operand1), pdn.Tensor(operand2)
    pdn_output: pdn.Tensor = pdn.matmul(pdn_operand1, pdn_operand2)
    np_output: np.ndarray = np.matmul(operand1, operand2)
    assert pdn_output.shape == np_output.shape
    assert pdn_output.dtype == np_output.dtype
    assert np.allclose(pdn_output.data, np_output, equal_nan=True)

