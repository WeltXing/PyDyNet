import math

import numpy as np
import pydynet as pdn
from pydynet import nn
import pydynet.nn.functional as F


def build_attention_mask(context_length: int):
    mask = np.full((context_length, context_length),
                   fill_value=-np.inf,
                   dtype=np.float32)
    mask = np.triu(mask, 1)
    return pdn.Tensor(mask, dtype=np.float32)


def patch_project(x: pdn.Tensor, kernel: pdn.Tensor):
    # Decompose images into 32x32 patches and multiply all patches by matrix.

    n, c, h, w = x.shape
    d, pc, ph, pw = kernel.shape
    p = pc * ph * pw
    gh = h // ph
    gw = w // pw

    assert c == pc and h % ph == 0 and w % pw == 0

    W = kernel.transpose(1, 2, 3, 0).reshape(p, d)
    x = x.reshape(n, c, gh, ph, gw, pw).transpose(0, 2, 4, 1, 3,
                                                  5).reshape(n, gh, gw, p)
    x = x @ W
    return x.reshape(n, gh * gw, d)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.n_dim = n_dim
        self.n_heads = n_heads
        self.head_dim = n_dim // n_heads

        self.QKV = nn.Linear(self.n_dim, self.n_dim * 3, dtype=np.float32)
        self.O = nn.Linear(self.n_dim, self.n_dim, dtype=np.float32)

    def forward(self, x, mask):
        B, L, _ = x.shape
        xq, xk, xv = pdn.split(self.QKV(x), 3, -1)
        xq = xq.reshape(B, L, self.n_heads, self.head_dim)
        xk = xk.reshape(B, L, self.n_heads, self.head_dim)
        xv = xv.reshape(B, L, self.n_heads, self.head_dim)

        xq, xkT = xq.transpose(0, 2, 1, 3), xk.transpose(0, 2, 3, 1)
        attention = xq @ xkT / math.sqrt(self.head_dim)

        if mask is not None:
            attention = attention + mask

        attention = F.softmax(attention, axis=-1)
        output = attention @ xv.transpose(0, 2, 1, 3)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.O(output)


class CLIPLayerNorm(nn.LayerNorm):

    def __init__(self,
                 normalized_shape,
                 eps=0.000001,
                 momentum=0.1,
                 device=None,
                 dtype=None):
        super().__init__(normalized_shape, eps, momentum, device, dtype)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = pdn.square(x - mean).mean(axis=-1, keepdims=True)
        x = (x - mean) / pdn.sqrt(var + self.eps) * self.scale + self.shift
        return x


class MLP(nn.Module):

    def __init__(self, d_in: int, d_proj: int):
        super().__init__()
        self.d_in = d_in
        self.d_proj = d_proj
        self.fc1 = nn.Linear(d_in, d_proj, dtype=np.float32)
        self.fc2 = nn.Linear(d_proj, d_in, dtype=np.float32)

    def forward(self, x):
        x = self.fc1(x)
        x = x * pdn.sigmoid(1.702 * x)
        return self.fc2(x)


class Transformer(nn.Module):

    def __init__(self, n_dim: int, n_head: int, mlp_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(n_dim, n_head)
        self.mlp = MLP(n_dim, mlp_dim)
        self.layer_norm1 = CLIPLayerNorm((n_dim, ), eps=1e-5, dtype=np.float32)
        self.layer_norm2 = CLIPLayerNorm((n_dim, ), eps=1e-5, dtype=np.float32)

    def forward(self, x, mask):
        x = x + self.mha(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class ImageEncoder(nn.Module):

    def __init__(self, n_dim, n_head, mlp_dim, kernel_size, n_layer,
                 final_dim):
        super().__init__()
        self.kernel = nn.Parameter(
            pdn.randn(n_dim, 3, kernel_size, kernel_size, dtype=np.float32))

        self.pre_norm = CLIPLayerNorm((n_dim, ), 1e-5, dtype=np.float32)
        self.transformers: list[Transformer] = nn.ModuleList(
            [Transformer(n_dim, n_head, mlp_dim) for _ in range(n_layer)])

        self.post_norm = CLIPLayerNorm((n_dim, ), 1e-5, dtype=np.float32)
        self.proj = nn.Linear(n_dim, final_dim, bias=False, dtype=np.float32)

    def forward(self, x, class_emb, position_emb):
        x = patch_project(x, self.kernel)
        x = pdn.concat([class_emb, x], axis=-2) + position_emb

        x = self.pre_norm(x)
        for model in self.transformers:
            x = model(x, None)

        x = self.post_norm(x[:, 0])
        return self.proj(x)


class TextEncoder(nn.Module):

    def __init__(self, n_dim, n_head, mlp_dim, n_layer, final_dim, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_dim, dtype=np.float32)
        self.transformers: list[Transformer] = nn.ModuleList(
            [Transformer(n_dim, n_head, mlp_dim) for _ in range(n_layer)])

        self.post_norm = CLIPLayerNorm((n_dim, ), 1e-5, dtype=np.float32)
        self.proj = nn.Linear(n_dim, final_dim, bias=False, dtype=np.float32)

    def forward(self, idx, position_emb):
        x = self.token_embed(idx) + position_emb
        mask = build_attention_mask(x.shape[1])

        for model in self.transformers:
            x = model(x, mask)

        x = self.post_norm(x)

        return self.proj(x[np.arange(x.shape[0]), x.xp.argmax(idx, axis=-1)])


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.class_embed = nn.Parameter(pdn.randn(1, 1, 768, dtype=np.float32))
        self.v_pos_emb = nn.Parameter(pdn.randn(50, 768, dtype=np.float32))
        self.t_pos_emb = nn.Parameter(pdn.randn(77, 512, dtype=np.float32))
        self.image_encoder = ImageEncoder(768, 12, 3072, 32, 12, 512)
        self.text_encoder = TextEncoder(512, 8, 2048, 12, 512, 49408)
        self.scale = 1

    def forward(self, img, idx):
        img_feature = self.image_encoder(img, self.class_embed, self.v_pos_emb)
        txt_feature = self.text_encoder(idx, self.t_pos_emb)

        norm_img = pdn.sqrt(pdn.square(img_feature).sum(1, keepdims=True))
        norm_txt = pdn.sqrt(pdn.square(txt_feature).sum(1, keepdims=True))

        img_feature = img_feature / norm_img
        txt_feature = txt_feature / norm_txt
        logits_per_image = self.scale * img_feature @ txt_feature.T
        return logits_per_image
