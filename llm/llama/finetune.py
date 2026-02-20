import argparse
import time

import numpy as np

import pydynet as pdn
import pydynet.optim as optim

from .io import load_model, save_finetuned_parameters
from .model import Llama
from .tokenizer import Tokenizer


def build_causal_training_pair(tokenizer: Tokenizer, text: str, max_seq_len: int):
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(token_ids) < 2:
        raise ValueError("Training text is too short after tokenization.")

    # +1 for shifted labels: [x0..xN-1] -> [x1..xN]
    token_ids = token_ids[:max_seq_len + 1]
    if len(token_ids) < 2:
        raise ValueError("Token sequence must contain at least 2 tokens.")

    input_ids = np.array([token_ids[:-1]], dtype=np.int64)
    target_ids = np.array([token_ids[1:]], dtype=np.int64)
    return input_ids, target_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Llama parameters")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--trainable", type=str, default="lm_head",
                        help="Comma-separated parameter name prefixes to train")
    parser.add_argument("--save", type=str, default="llm/llama/data/finetuned_params.npz")
    args = parser.parse_args()

    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 1024
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

    if args.cuda and pdn.cuda.is_available():
        model = model.to('cuda:0')

    prefixes = tuple(p.strip() for p in args.trainable.split(',') if p.strip())
    trainable_count, frozen_count = model.set_trainable_parameters(prefixes)
    print(f"Trainable params: {trainable_count}, Frozen params: {frozen_count}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    input_ids, target_ids = build_causal_training_pair(tokenizer, args.text, max_seq_len)

    start = time.time()
    for step in range(1, args.steps + 1):
        loss = model.finetune_step(input_ids, target_ids, optimizer)
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step:04d}, loss={loss:.6f}")

    elapsed = time.time() - start
    save_finetuned_parameters(model, args.save)
    print(f"Saved finetuned params to {args.save}")
    print(f"Elapsed: {elapsed:.2f}s")
