import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pydynet.optim as optim
from llm.llama.model import Llama


np.random.seed(7)


def _build_tiny_llama(dtype=np.float32):
    return Llama(
        vocab_size=16,
        embed_dim=8,
        n_heads=2,
        ffn_dim=16,
        max_seq_len=8,
        max_batch_size=1,
        n_layers=2,
        dtype=dtype,
    )


def test_forward_logits_shape_for_training():
    model = _build_tiny_llama()
    input_ids = np.array([[1, 3, 5, 7]], dtype=np.int64)
    logits = model.forward_logits(input_ids)
    assert logits.shape == (1, 4, 16)


def test_set_trainable_parameters_freezes_non_selected_prefixes():
    model = _build_tiny_llama()
    trainable_count, frozen_count = model.set_trainable_parameters(("lm_head", ))

    assert trainable_count > 0
    assert frozen_count > 0

    for name, param in model._parameters.items():
        if name.startswith("lm_head"):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_finetune_step_updates_selected_parameters_only():
    model = _build_tiny_llama(dtype=np.float64)
    model.set_trainable_parameters(("lm_head", ))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    target_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)

    lm_head_before = model.lm_head.weight.numpy()
    q_weight_before = model.layers[0].attention.Q.weight.numpy()

    loss = model.finetune_step(input_ids, target_ids, optimizer)

    assert np.isfinite(loss)
    assert not np.allclose(model.lm_head.weight.numpy(), lm_head_before)
    assert np.allclose(model.layers[0].attention.Q.weight.numpy(), q_weight_before)
