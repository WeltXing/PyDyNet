import os, json, urllib, zipfile
import urllib.request
from PIL import Image

import numpy as np
import pydynet as pdn
import pydynet.nn.functional as F

from .tokenizer import SimpleTokenizer
from .model import CLIP


def download(url: str, filename: str, chunk_size: int = 10**6) -> None:
    # Create directories if they don't exist yet
    directories = os.path.dirname(filename)
    if directories:
        os.makedirs(directories, exist_ok=True)

    # Download the file
    with urllib.request.urlopen(url) as response:
        total = int(response.info()["Content-Length"])

        buf = b""
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            buf += data
            print(f"Downloading {filename} {len(buf) / total * 100:.2f} %")

    # Write the downloaded data to the file
    with open(filename, "wb") as f:
        f.write(buf)


def load_zip(path: str):
    files = {}

    with zipfile.ZipFile(path) as z:
        for file_info in z.infolist():
            with z.open(file_info) as f:
                path = file_info.filename
                files[path] = f.read()

    return files


class Params:

    def __init__(self, name: str, download_root: str = None) -> None:
        assert name == "ViT-B/32", f"Model {name} not supported yet. Only ViT-B-32 currently supported."

        model_urls = {
            "RN50":
            "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
            "RN101":
            "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
            "RN50x4":
            "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
            "RN50x16":
            "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
            "RN50x64":
            "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
            "ViT-B/32":
            "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "ViT-B/16":
            "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
            "ViT-L/14":
            "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "ViT-L/14@336px":
            "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        }

        model_url = model_urls[name]

        name = name.replace("/", "-")

        if download_root is None:
            download_root = os.path.expanduser(f"~/.cache/clip")
            download_root = os.environ.get("CLIP_DIR", download_root)

        model_path = os.path.join(download_root, f"{name}.pt")

        if not os.path.isfile(model_path):
            print(f"Downloading {model_path} from {model_url}")
            download(model_url, model_path)

        self.files = load_zip(model_path)

        with open(f"{download_root}/{name}.json") as f:
            self.info = json.load(f)

    def get_int(self, name: str) -> int:
        info = self.info[name]

        value: int = info["value"]

        return value

    def __getitem__(self, name: str):
        info = self.info[name]

        path = info["path"]
        dtype = info["dtype"]
        shape = info["shape"]
        start = info["start"]
        end = info["end"]

        assert dtype in ["float16", "float32"]

        data = self.files[path][start:end]

        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        arr = arr.astype(np.float32)

        return arr


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
                f"Input {texts[i]} is too long for context length {context_length}"
            )

        result[i, :len(tokens)] = tokens

    return result


def preprocess(image: Image.Image, image_size: int = 224):
    # Scale image such that length of smaller side is 224
    width, height = image.size
    scale = image_size / min(width, height)
    width = int(scale * width)
    height = int(scale * height)
    # Some Pillow versions have different interface
    if hasattr(Image, "Resampling"):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    else:
        image = image.resize((width, height), Image.BICUBIC)

    # Crop center
    x0 = round((width - image_size) / 2)
    y0 = round((height - image_size) / 2)
    x1 = x0 + image_size
    y1 = y0 + image_size
    image = image.crop((x0, y0, x1, y1))

    image = image.convert("RGB")

    # Normalize
    x = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = (x - mean) / std

    x = x.transpose(2, 0, 1)

    return pdn.Tensor(x, copy=None)


@pdn.no_grad()
def load_model(model: CLIP, param: Params):

    # with pdn.no_grad():
    model.scale = pdn.exp(param["logit_scale"].astype(np.float32))
    model.class_embed.data[0, 0] = param["visual.class_embedding"]
    model.v_pos_emb.data[...] = param["visual.positional_embedding"]
    model.t_pos_emb.data[...] = param["positional_embedding"]

    model.image_encoder.kernel.data[...] = param["visual.conv1.weight"]
    model.image_encoder.pre_norm.scale[...] = param["visual.ln_pre.weight"]
    model.image_encoder.pre_norm.shift[...] = param["visual.ln_pre.bias"]
    model.image_encoder.post_norm.scale[...] = param["visual.ln_post.weight"]
    model.image_encoder.post_norm.shift[...] = param["visual.ln_post.bias"]

    model.image_encoder.proj.weight[...] = param["visual.proj"]

    model.text_encoder.token_embed.weight[
        ...] = param["token_embedding.weight"]
    model.text_encoder.post_norm.scale[...] = param["ln_final.weight"]
    model.text_encoder.post_norm.shift[...] = param["ln_final.bias"]
    model.text_encoder.proj.weight[...] = param["text_projection"]

    prefix = "transformer.resblocks."
    for i in range(12):
        (
            model.image_encoder.transformers[i].mha.QKV.weight.data[...],
            model.image_encoder.transformers[i].mha.QKV.bias.data[...],
            model.image_encoder.transformers[i].mha.O.weight.data[...],
            model.image_encoder.transformers[i].mha.O.bias.data[...],
            model.image_encoder.transformers[i].layer_norm1.scale.data[...],
            model.image_encoder.transformers[i].layer_norm1.shift.data[...],
            model.image_encoder.transformers[i].layer_norm2.scale.data[...],
            model.image_encoder.transformers[i].layer_norm2.shift.data[...],
            model.image_encoder.transformers[i].mlp.fc1.weight.data[...],
            model.image_encoder.transformers[i].mlp.fc1.bias.data[...],
            model.image_encoder.transformers[i].mlp.fc2.weight.data[...],
            model.image_encoder.transformers[i].mlp.fc2.bias.data[...],
            model.text_encoder.transformers[i].mha.QKV.weight.data[...],
            model.text_encoder.transformers[i].mha.QKV.bias.data[...],
            model.text_encoder.transformers[i].mha.O.weight.data[...],
            model.text_encoder.transformers[i].mha.O.bias.data[...],
            model.text_encoder.transformers[i].layer_norm1.scale.data[...],
            model.text_encoder.transformers[i].layer_norm1.shift.data[...],
            model.text_encoder.transformers[i].layer_norm2.scale.data[...],
            model.text_encoder.transformers[i].layer_norm2.shift.data[...],
            model.text_encoder.transformers[i].mlp.fc1.weight.data[...],
            model.text_encoder.transformers[i].mlp.fc1.bias.data[...],
            model.text_encoder.transformers[i].mlp.fc2.weight.data[...],
            model.text_encoder.transformers[i].mlp.fc2.bias.data[...],
        ) = (
            param["visual." + prefix + f"{i}.attn.in_proj_weight"].T,
            param["visual." + prefix + f"{i}.attn.in_proj_bias"],
            param["visual." + prefix + f"{i}.attn.out_proj.weight"].T,
            param["visual." + prefix + f"{i}.attn.out_proj.bias"],
            param["visual." + prefix + f"{i}.ln_1.weight"],
            param["visual." + prefix + f"{i}.ln_1.bias"],
            param["visual." + prefix + f"{i}.ln_2.weight"],
            param["visual." + prefix + f"{i}.ln_2.bias"],
            param["visual." + prefix + f"{i}.mlp.c_fc.weight"].T,
            param["visual." + prefix + f"{i}.mlp.c_fc.bias"],
            param["visual." + prefix + f"{i}.mlp.c_proj.weight"].T,
            param["visual." + prefix + f"{i}.mlp.c_proj.bias"],
            param[prefix + f"{i}.attn.in_proj_weight"].T,
            param[prefix + f"{i}.attn.in_proj_bias"],
            param[prefix + f"{i}.attn.out_proj.weight"].T,
            param[prefix + f"{i}.attn.out_proj.bias"],
            param[prefix + f"{i}.ln_1.weight"],
            param[prefix + f"{i}.ln_1.bias"],
            param[prefix + f"{i}.ln_2.weight"],
            param[prefix + f"{i}.ln_2.bias"],
            param[prefix + f"{i}.mlp.c_fc.weight"].T,
            param[prefix + f"{i}.mlp.c_fc.bias"],
            param[prefix + f"{i}.mlp.c_proj.weight"].T,
            param[prefix + f"{i}.mlp.c_proj.bias"],
        )
    return model


image = preprocess(Image.open("llm/clip/picture.png"))[np.newaxis, :, :, :]
text = tokenize(["a fish", "a dog", "a cat"])
clip = load_model(CLIP(), Params("ViT-B/32", download_root='llm/clip/data'))

with pdn.no_grad():
    logits_per_image = clip(image, text)
    probs = F.softmax(logits_per_image, axis=-1)
    print("Label probs:", probs.numpy()[0])
