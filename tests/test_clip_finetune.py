import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pydynet as pdn
import pydynet.optim as optim
from llm.clip.io import load_finetuned_parameters, save_finetuned_parameters
from llm.clip.model import CLIP


np.random.seed(11)


def _tiny_clip():
    return CLIP(
        image_dim=8,
        image_heads=2,
        image_mlp_dim=16,
        image_patch=2,
        image_layers=1,
        text_dim=8,
        text_heads=2,
        text_mlp_dim=16,
        text_layers=1,
        final_dim=8,
        vocab_size=32,
        vision_tokens=5,
        text_tokens=6,
    )


def test_clip_forward_shape():
    model = _tiny_clip()
    image = pdn.Tensor(np.random.randn(1, 3, 4, 4).astype(np.float32))
    text = np.random.randint(0, 32, size=(3, 6), dtype=np.int64)

    logits = model(image, text)
    assert logits.shape == (1, 3)


def test_clip_set_trainable_parameters_by_prefix():
    model = _tiny_clip()
    trainable_count, frozen_count = model.set_trainable_parameters(("text_encoder.proj", ))

    assert trainable_count > 0
    assert frozen_count > 0

    for name, param in model._parameters.items():
        if name.startswith("text_encoder.proj"):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_clip_finetune_step_updates_selected_parameters_only():
    model = _tiny_clip()
    model.set_trainable_parameters(("text_encoder.proj", ))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    image = pdn.Tensor(np.random.randn(1, 3, 4, 4).astype(np.float32))
    text = np.random.randint(0, 32, size=(3, 6), dtype=np.int64)
    target = np.array([1], dtype=np.int64)

    text_proj_before = model.text_encoder.proj.weight.numpy()
    image_proj_before = model.image_encoder.proj.weight.numpy()

    loss = model.finetune_step(image, text, target, optimizer)

    assert np.isfinite(loss)
    assert not np.allclose(model.text_encoder.proj.weight.numpy(), text_proj_before)
    assert np.allclose(model.image_encoder.proj.weight.numpy(), image_proj_before)


def test_clip_finetuned_io_roundtrip(tmp_path):
    model = _tiny_clip()
    model.set_trainable_parameters(("text_encoder.proj", ))

    original = model.text_encoder.proj.weight.numpy()
    model.text_encoder.proj.weight.data += 0.123

    save_path = tmp_path / "clip_ft.npz"
    save_finetuned_parameters(model, str(save_path))

    restored = _tiny_clip()
    restored.set_trainable_parameters(("text_encoder.proj", ))
    load_finetuned_parameters(restored, str(save_path))

    assert not np.allclose(restored.text_encoder.proj.weight.numpy(), original)
    assert np.allclose(restored.text_encoder.proj.weight.numpy(),
                       model.text_encoder.proj.weight.numpy())
