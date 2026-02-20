import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pydynet as pdn


np.random.seed(1)


def test_unary_function_forward_matches_numpy():
    x_np = np.random.uniform(0.5, 2.0, size=(3, 4)).astype(np.float64)
    x = pdn.Tensor(x_np)

    pairs = [
        (pdn.abs, np.abs),
        (pdn.exp, np.exp),
        (pdn.log, np.log),
        (pdn.sign, np.sign),
        (pdn.sigmoid, lambda z: 1.0 / (1.0 + np.exp(-z))),
        (pdn.tanh, np.tanh),
        (pdn.sqrt, np.sqrt),
        (pdn.square, np.square),
    ]

    for pdn_func, np_func in pairs:
        pdn_out = pdn_func(x)
        np_out = np_func(x_np)
        assert pdn_out.shape == np_out.shape
        assert np.allclose(pdn_out.data, np_out, atol=1e-6, rtol=1e-6)


def test_reduce_function_forward_matches_numpy():
    x_np = np.random.randn(2, 3, 4).astype(np.float64)
    x = pdn.Tensor(x_np)

    test_cases = [
        (lambda t: pdn.sum(t), lambda a: np.sum(a)),
        (lambda t: pdn.mean(t), lambda a: np.mean(a)),
        (lambda t: pdn.sum(t, axis=1), lambda a: np.sum(a, axis=1)),
        (lambda t: pdn.mean(t, axis=(0, 2), keepdims=True),
         lambda a: np.mean(a, axis=(0, 2), keepdims=True)),
        (lambda t: pdn.max(t, axis=2), lambda a: np.max(a, axis=2)),
        (lambda t: pdn.min(t, axis=0), lambda a: np.min(a, axis=0)),
        (lambda t: pdn.argmax(t, axis=1), lambda a: np.argmax(a, axis=1)),
        (lambda t: pdn.argmin(t, axis=2), lambda a: np.argmin(a, axis=2)),
    ]

    for pdn_func, np_func in test_cases:
        pdn_out = pdn_func(x)
        np_out = np_func(x_np)
        assert pdn_out.shape == np_out.shape
        assert np.allclose(pdn_out.data, np_out)


def test_shape_manipulation_matches_numpy():
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    x = pdn.Tensor(x_np)

    reshape_out = pdn.reshape(x, (4, 6))
    assert np.array_equal(reshape_out.data, x_np.reshape(4, 6))

    transpose_out = pdn.transpose(x, (2, 0, 1))
    assert np.array_equal(transpose_out.data, x_np.transpose(2, 0, 1))

    swapaxes_out = pdn.swapaxes(x, 0, 2)
    assert np.array_equal(swapaxes_out.data, np.swapaxes(x_np, 0, 2))

    unsqueeze_out = pdn.unsqueeze(x, (0, 2))
    assert np.array_equal(unsqueeze_out.data, np.expand_dims(np.expand_dims(x_np, 0), 2))

    squeeze_in = pdn.Tensor(np.ones((1, 2, 1, 3), dtype=np.float64))
    squeeze_out = pdn.squeeze(squeeze_in, axis=(0, 2))
    assert np.array_equal(squeeze_out.data, np.squeeze(squeeze_in.data, axis=(0, 2)))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_split_and_concat_roundtrip(axis):
    x_np = np.random.randn(4, 6, 8).astype(np.float64)
    x = pdn.Tensor(x_np)

    pieces = pdn.split(x, 2, axis=axis)
    assert len(pieces) == 2

    merged = pdn.concat(pieces, axis=axis)
    assert np.allclose(merged.data, x_np)


def test_concat_backward_distributes_gradient():
    a_np = np.random.randn(2, 3).astype(np.float64)
    b_np = np.random.randn(2, 2).astype(np.float64)

    a = pdn.Tensor(a_np, requires_grad=True)
    b = pdn.Tensor(b_np, requires_grad=True)

    y = pdn.concat([a, b], axis=1).sum()
    y.backward()

    assert np.array_equal(a.grad, np.ones_like(a_np))
    assert np.array_equal(b.grad, np.ones_like(b_np))


def test_mean_backward_with_axis_and_keepdims():
    x_np = np.random.randn(2, 3, 4).astype(np.float64)
    x = pdn.Tensor(x_np, requires_grad=True)

    y = pdn.mean(x, axis=1, keepdims=True).sum()
    y.backward()

    expected = np.ones_like(x_np) / x_np.shape[1]
    assert np.allclose(x.grad, expected)
