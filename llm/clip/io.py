import json
import os
import urllib.request
import zipfile

import numpy as np
import pydynet as pdn

from .model import CLIP


def download(url: str, filename: str, chunk_size: int = 10**6) -> None:
    directories = os.path.dirname(filename)
    if directories:
        os.makedirs(directories, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total = int(response.info()["Content-Length"])

        buf = b""
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            buf += data
            print(f"Downloading {filename} {len(buf) / total * 100:.2f} %")

    with open(filename, "wb") as f:
        f.write(buf)


def load_zip(path: str):
    files = {}
    with zipfile.ZipFile(path) as z:
        for file_info in z.infolist():
            with z.open(file_info) as f:
                files[file_info.filename] = f.read()
    return files


class Params:

    def __init__(self, name: str, download_root: str = None) -> None:
        assert name == "ViT-B/32", f"Model {name} not supported yet. Only ViT-B-32 currently supported."

        model_urls = {
            "ViT-B/32":
            "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        }

        model_url = model_urls[name]
        name = name.replace("/", "-")

        if download_root is None:
            download_root = os.path.expanduser("~/.cache/clip")
            download_root = os.environ.get("CLIP_DIR", download_root)

        model_path = os.path.join(download_root, f"{name}.pt")

        if not os.path.isfile(model_path):
            print(f"Downloading {model_path} from {model_url}")
            download(model_url, model_path)

        self.files = load_zip(model_path)

        with open(f"{download_root}/{name}.json") as f:
            self.info = json.load(f)

    def __getitem__(self, name: str):
        info = self.info[name]
        data = self.files[info["path"]][info["start"]:info["end"]]
        arr = np.frombuffer(data, dtype=info["dtype"]).reshape(info["shape"])
        return arr.astype(np.float32)


@pdn.no_grad()
def load_model(model: CLIP, param: Params):
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

    model.text_encoder.token_embed.weight[...] = param["token_embedding.weight"]
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


@pdn.no_grad()
def save_finetuned_parameters(model: CLIP, output_path: str):
    params = {}
    for name, param in model._parameters.items():
        if param.requires_grad:
            params[name] = param.numpy()
    np.savez(output_path, **params)


@pdn.no_grad()
def load_finetuned_parameters(model: CLIP, finetuned_path: str):
    weights = np.load(finetuned_path)
    for name, param in model._parameters.items():
        if name in weights:
            param.data[...] = weights[name]
    return model
