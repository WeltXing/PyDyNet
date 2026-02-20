import argparse

from PIL import Image
import numpy as np
import pydynet.optim as optim

from .data import preprocess, tokenize
from .io import Params, load_model, save_finetuned_parameters
from .model import CLIP


def parse_labels(labels: str):
    vals = [x.strip() for x in labels.split(',') if x.strip()]
    if len(vals) < 2:
        raise ValueError('Need at least 2 labels for classification fine-tuning.')
    return vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune CLIP parameters')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True,
                        help='Comma-separated candidate labels')
    parser.add_argument('--target', type=int, required=True,
                        help='Index in labels that is the ground-truth class')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--trainable', type=str, default='text_encoder.proj,image_encoder.proj',
                        help='Comma-separated parameter name prefixes to train')
    parser.add_argument('--save', type=str, default='llm/clip/data/finetuned_clip_params.npz')
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    if args.target < 0 or args.target >= len(labels):
        raise ValueError('target index out of range for labels.')

    image = preprocess(Image.open(args.image))[np.newaxis, :, :, :]
    text = tokenize(labels)
    target = np.array([args.target], dtype=np.int64)

    model = load_model(CLIP(), Params('ViT-B/32', download_root='llm/clip/data'))

    prefixes = tuple(p.strip() for p in args.trainable.split(',') if p.strip())
    trainable_count, frozen_count = model.set_trainable_parameters(prefixes)
    print(f'Trainable params: {trainable_count}, Frozen params: {frozen_count}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        loss = model.finetune_step(image, text, target, optimizer)
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f'step={step:04d}, loss={loss:.6f}')

    save_finetuned_parameters(model, args.save)
    print(f'Saved finetuned params to {args.save}')
