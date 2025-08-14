import math
import numpy as np

import pydynet as pdn
from pydynet.tensor import Tensor
import pydynet.nn as nn
import pydynet.nn.functional as F


def compute_cos_sin_cache(head_dim: int,
                          max_seq_len: int,
                          base: int = 10000,
                          dtype=None):
    inv_freq = 1.0 / (base**(np.arange(0, head_dim, 2)[:(head_dim // 2)] /
                             head_dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq).astype(dtype)

    return Tensor(np.cos(freqs)), Tensor(np.sin(freqs))


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cos, freqs_sin):
    xqri = xq.reshape(*(xq.shape[:-1] + (-1, 2)))
    xkri = xk.reshape(*(xk.shape[:-1] + (-1, 2)))

    xq_r, xq_i = xqri[..., 0], xqri[..., 1]
    xk_r, xk_i = xkri[..., 0], xkri[..., 1]

    freqs_cos = pdn.unsqueeze(freqs_cos, axis=-2)
    freqs_sin = pdn.unsqueeze(freqs_sin, axis=-2)

    # Apply rotation using real numbers.
    xq_out_r = pdn.unsqueeze(xq_r * freqs_cos - xq_i * freqs_sin, -1)
    xq_out_i = pdn.unsqueeze(xq_r * freqs_sin + xq_i * freqs_cos, -1)
    xk_out_r = pdn.unsqueeze(xk_r * freqs_cos - xk_i * freqs_sin, -1)
    xk_out_i = pdn.unsqueeze(xk_r * freqs_sin + xk_i * freqs_cos, -1)

    # Flatten last two dimensions.
    xq_out = pdn.concat([xq_out_r, xq_out_i], axis=-1)
    xk_out = pdn.concat([xk_out_r, xk_out_i], axis=-1)
    xq_out = xq_out.reshape(*(xq_out.shape[:-2] + (-1, )))
    xk_out = xk_out.reshape(*(xk_out.shape[:-2] + (-1, )))
    return xq_out, xk_out


class FeedForward(nn.Module):

    def __init__(self, dim, up_dim, dtype=None):
        super().__init__()
        self.dim, self.up_dim = dim, up_dim
        self.up = nn.Linear(dim, up_dim, bias=False, dtype=dtype)
        self.gate = nn.Linear(dim, up_dim, bias=False, dtype=dtype)
        self.down = nn.Linear(up_dim, dim, bias=False, dtype=dtype)

    def forward(self, x):
        swish, x_V = F.silu(self.gate(x)), self.up(x)
        return self.down(swish * x_V)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        max_batch_size: int = None,
        dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.Q = nn.Linear(self.dim, self.dim, bias=False, dtype=dtype)
        self.K = nn.Linear(self.dim, self.dim, bias=False, dtype=dtype)
        self.V = nn.Linear(self.dim, self.dim, bias=False, dtype=dtype)
        self.O = nn.Linear(self.dim, self.dim, bias=False, dtype=dtype)

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size if max_batch_size is not None else 1

        self.cache_k = nn.Parameter(pdn.special.zeros(
            (self.max_batch_size, max_seq_len, self.n_heads, self.head_dim),
            dtype=dtype),
                                    requires_grad=False)
        self.cache_v = nn.Parameter(pdn.special.zeros(
            (self.max_batch_size, max_seq_len, self.n_heads, self.head_dim),
            dtype=dtype),
                                    requires_grad=False)

    def __call__(self, x, start_pos: int, mask, freqs_cos, freqs_sin):
        B, L, _ = x.shape
        xq, xk, xv = (
            self.Q(x).reshape(B, L, self.n_heads, self.head_dim),
            self.K(x).reshape(B, L, self.n_heads, self.head_dim),
            self.V(x).reshape(B, L, self.n_heads, self.head_dim),
        )

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if not self._train:
            self.cache_k[:B, start_pos:start_pos + L] = xk
            self.cache_v[:B, start_pos:start_pos + L] = xv

            xk = self.cache_k[:B, :start_pos + L]
            xv = self.cache_v[:B, :start_pos + L]

        xq, xkT = xq.transpose(0, 2, 1, 3), xk.transpose(0, 2, 3, 1)
        attention = xq @ xkT / math.sqrt(self.head_dim)

        if mask is not None:
            attention = attention + mask
        attention = F.softmax(attention, axis=-1)
        output = attention @ xv.transpose(0, 2, 1, 3)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.O(output)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        max_batch_size: int = None,
        dtype=None,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, max_seq_len, max_batch_size,
                                   dtype)
        self.ffn = FeedForward(dim, ffn_dim, dtype)
        self.input_norm = nn.RMSNorm(dim, dtype=dtype)
        self.post_attn_norm = nn.RMSNorm(dim, dtype=dtype)

    def forward(self, x, start_pos: int, mask, freqs_cos, freqs_sin):
        norm_x = self.input_norm(x)

        h1 = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        norm_z = self.post_attn_norm(z)
        h2 = self.ffn(norm_z)
        return z + h2


class Llama(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        n_heads,
        ffn_dim: int,
        max_seq_len: int,
        max_batch_size: int = None,
        n_layers: int = 6,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.n_layers = n_layers

        self.tok_embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype)
        freqs_cos, freqs_sin = compute_cos_sin_cache(embed_dim // n_heads,
                                                     max_seq_len,
                                                     dtype=dtype)

        self.freqs_cos = nn.parameter.Parameter(freqs_cos, False)
        self.freqs_sin = nn.parameter.Parameter(freqs_sin, False)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, max_seq_len,
                             max_batch_size, dtype)
            for _ in range(self.n_layers)
        ])

        self.norm = nn.RMSNorm(embed_dim, dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, dtype=dtype)

    def forward(self, input_ids, start_pos: int):
        L = input_ids.shape[-1]
        h = self.tok_embedding(input_ids)

        freqs_cos = self.freqs_cos[start_pos:start_pos + L]
        freqs_sin = self.freqs_sin[start_pos:start_pos + L]

        mask = None
        if L > 1:
            mask = np.triu(np.full((L, L), float("-inf")), k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)
            mask = pdn.Tensor(mask, device=h.device, dtype=h.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        logit = self.lm_head(self.norm(h)[:, [-1], :])
        return logit

    def generate(self, input_ids, max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, True)
            yield next_id
