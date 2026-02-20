import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pydynet as pdn

np.random.seed(0)
random.seed(0)


def _assert_allclose(actual, expected, atol=1e-6, rtol=1e-6):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol)


def test_backward_scalar_polynomial():
    x = pdn.Tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x - 1
    y.backward()
    _assert_allclose(x.grad, np.array(7.0))


def test_backward_broadcast_add():
    x_np = np.random.randn(2, 3).astype(np.float64)
    b_np = np.random.randn(1, 3).astype(np.float64)

    x = pdn.Tensor(x_np, requires_grad=True)
    b = pdn.Tensor(b_np, requires_grad=True)

    y = (x + b).sum()
    y.backward()

    _assert_allclose(x.grad, np.ones_like(x_np))
    _assert_allclose(b.grad, np.full_like(b_np, x_np.shape[0]))


def test_backward_matmul_sum():
    x_np = np.random.randn(2, 3).astype(np.float64)
    w_np = np.random.randn(3, 4).astype(np.float64)

    x = pdn.Tensor(x_np, requires_grad=True)
    w = pdn.Tensor(w_np, requires_grad=True)

    y = pdn.matmul(x, w).sum()
    y.backward()

    expected_x_grad = np.ones((2, 4), dtype=np.float64) @ w_np.T
    expected_w_grad = x_np.T @ np.ones((2, 4), dtype=np.float64)

    _assert_allclose(x.grad, expected_x_grad)
    _assert_allclose(w.grad, expected_w_grad)


def test_backward_retain_graph_twice_accumulates_grad():
    x = pdn.Tensor(2.0, requires_grad=True)
    y = x * x

    y.backward(retain_graph=True)
    first_grad = np.array(x.grad, copy=True)
    y.backward()

    _assert_allclose(first_grad, np.array(4.0))
    _assert_allclose(x.grad, np.array(8.0))


def test_backward_on_non_scalar_raises():
    x = pdn.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(ValueError, match="scalar"):
        x.backward()
