import argparse

from PIL import Image
import numpy as np
import pydynet as pdn
import pydynet.nn.functional as F

from .data import preprocess, tokenize
from .io import Params, load_finetuned_parameters, load_model
from .model import CLIP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP image-text inference')
    parser.add_argument('--image', type=str, default='llm/clip/picture.png')
    parser.add_argument('--labels', type=str, default='a fish,a dog,a cat')
    parser.add_argument('--finetuned', type=str, default=None,
                        help='Optional finetuned parameter file (.npz)')
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(',') if x.strip()]

    image = preprocess(Image.open(args.image))[np.newaxis, :, :, :]
    text = tokenize(labels)

    clip = load_model(CLIP(), Params('ViT-B/32', download_root='llm/clip/data'))
    if args.finetuned is not None:
        clip = load_finetuned_parameters(clip, args.finetuned)

    with pdn.no_grad():
        logits_per_image = clip(image, text)
        probs = F.softmax(logits_per_image, axis=-1)
        print('Labels:', labels)
        print('Label probs:', probs.numpy()[0])
