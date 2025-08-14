import sys, time, argparse

sys.path.append('../pydynet')
from llama.tokenizer import Tokenizer
from llama.model import Llama

import pydynet as pdn
import numpy as np


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
            llama.layers[i].ffn.down.weight[...],
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prompt input, e.g. There was a boy")
    parser.add_argument("--prompt", type=str, default='There was a boy')
    parser.add_argument("--cuda", action='store_true')
    args = parser.parse_args()

    dim: int = 288  # D
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 32000  # VS
    max_seq_len: int = 1024  # M
    max_new_tokens: int = 1024
    max_batch_size: int = 1
    datatype = np.float32

    tokenizer = Tokenizer("llama/data/tokenizer.model.np")
    model = load_model(
        Llama(vocab_size,
              dim,
              n_heads,
              768,
              max_seq_len,
              max_batch_size,
              n_layers,
              dtype=datatype), "llama/data/stories15M.model.npz")

    # If cuda is available
    if args.cuda and pdn.cuda.is_available():
        model: Llama = model.to('cuda:2')

    model.eval()
    with pdn.no_grad():
        print(f"\n{args.prompt}", end="")
        input_ids = np.array([tokenizer.encode(args.prompt)])

        _, L = input_ids.shape
        start = time.time()
        for id in model.generate(input_ids, max_new_tokens):
            L += 1
            output_id = id[0].numpy().tolist()

            if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
                break
            print(tokenizer.decode(output_id), end="")
            sys.stdout.flush()
        elapsed = time.time() - start
        print(
            f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s"
        )
