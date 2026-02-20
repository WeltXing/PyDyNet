from PIL import Image
import numpy as np
import pydynet as pdn

from .tokenizer import SimpleTokenizer


def tokenize(texts: list[str], context_length: int = 77):
    tokenizer = SimpleTokenizer()

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token]
                  for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(
                f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = tokens

    return result


def preprocess(image: Image.Image, image_size: int = 224):
    width, height = image.size
    scale = image_size / min(width, height)
    width = int(scale * width)
    height = int(scale * height)
    if hasattr(Image, "Resampling"):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    else:
        image = image.resize((width, height), Image.BICUBIC)

    x0 = round((width - image_size) / 2)
    y0 = round((height - image_size) / 2)
    x1 = x0 + image_size
    y1 = y0 + image_size
    image = image.crop((x0, y0, x1, y1)).convert("RGB")

    x = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = (x - mean) / std

    x = x.transpose(2, 0, 1)
    return pdn.Tensor(x, copy=None)
