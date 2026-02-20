import numpy as np

import pydynet as pdn

from .model import Llama


@pdn.no_grad()
def load_model(llama: Llama, model_path: str) -> Llama:
    weight = np.load(model_path)

    llama.tok_embedding.weight.data[...] = weight['model.embed_tokens.weight']
    llama.lm_head.weight.data[...] = weight['lm_head.weight'].T

    for i in range(llama.n_layers):
        (
            llama.layers[i].attention.Q.weight.data[...],
            llama.layers[i].attention.K.weight.data[...],
            llama.layers[i].attention.V.weight.data[...],
            llama.layers[i].attention.O.weight.data[...],
            llama.layers[i].ffn.up.weight.data[...],
            llama.layers[i].ffn.gate.weight.data[...],
            llama.layers[i].ffn.down.weight.data[...],
            llama.layers[i].input_norm.weight.data[...],
            llama.layers[i].post_attn_norm.weight.data[...],
        ) = (
            weight[f'model.layers.{i}.self_attn.q_proj.weight'].T,
            weight[f'model.layers.{i}.self_attn.k_proj.weight'].T,
            weight[f'model.layers.{i}.self_attn.v_proj.weight'].T,
            weight[f'model.layers.{i}.self_attn.o_proj.weight'].T,
            weight[f'model.layers.{i}.mlp.up_proj.weight'].T,
            weight[f'model.layers.{i}.mlp.gate_proj.weight'].T,
            weight[f'model.layers.{i}.mlp.down_proj.weight'].T,
            weight[f'model.layers.{i}.input_layernorm.weight'],
            weight[f'model.layers.{i}.post_attention_layernorm.weight'],
        )

    llama.norm.weight.data[...] = weight['model.norm.weight']
    return llama


@pdn.no_grad()
def save_finetuned_parameters(model: Llama, output_path: str):
    params = {}
    for name, param in model._parameters.items():
        if param.requires_grad:
            params[name] = param.numpy()
    np.savez(output_path, **params)


@pdn.no_grad()
def load_finetuned_parameters(model: Llama, finetuned_path: str):
    weights = np.load(finetuned_path)
    for name, param in model._parameters.items():
        if name in weights:
            param.data[...] = weights[name]
    return model
