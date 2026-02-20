import sys, time, argparse
from .tokenizer import Tokenizer
from .model import Llama
from .io import load_model, load_finetuned_parameters

import pydynet as pdn
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prompt input, e.g. There was a boy")
    parser.add_argument("--prompt", type=str, default='There was a boy')
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--finetuned", type=str, default=None,
                        help="Optional finetuned parameter file (.npz)")
    args = parser.parse_args()

    dim: int = 288  # D
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 32000  # VS
    max_seq_len: int = 1024  # M
    max_new_tokens: int = 1024
    max_batch_size: int = 1
    datatype = np.float32

    tokenizer = Tokenizer("llm/llama/data/tokenizer.model.np")
    model = load_model(
        Llama(vocab_size,
              dim,
              n_heads,
              768,
              max_seq_len,
              max_batch_size,
              n_layers,
              dtype=datatype), "llm/llama/data/stories15M.model.npz")

    if args.finetuned is not None:
        model = load_finetuned_parameters(model, args.finetuned)

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
