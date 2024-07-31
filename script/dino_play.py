import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch import ByteTensor, FloatTensor
from torchvision.io.image import ImageReadMode, read_image, decode_jpeg, read_file
from torchvision.transforms.v2 import Resize, CenterCrop, Compose, Normalize
from dino_embed.chunk import chunk

@dataclass
class Args:
    in_dir: Path
    out_dir: Path

jpeg_suffices = {'.jpg', '.jpeg'}
def load_image(p: Path, device=torch.device('cpu')) -> None:
    if p.suffix in jpeg_suffices and device.type == 'cuda':
        data: ByteTensor = read_file(str(p))
        # NOTE: we deliberately forego apply_exif_orientation, because nvjpeg doesn't support it.
        #       instead we will, uh, hope none of these images rely on EXIF for orientation.
        img: ByteTensor = decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
    else:
        img: ByteTensor = read_image(str(p), mode=ImageReadMode.RGB, apply_exif_orientation=True).to(device)
    return img

img_suffices = {*jpeg_suffices, '.png'}
def main(args: Args) -> None:
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # we could make this lazier, but I'd rather have a progress bar
    candidates: list[Path] = [p for p in args.in_dir.rglob('**/*') if p.suffix in img_suffices]
    for batch in tqdm(chunk(candidates, 8)):
        tensors: list[FloatTensor] = [load_image(p, device=device) for p in batch]
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dino Play')
    parser.add_argument('--in-dir', type=Path, default=Path('in'))
    parser.add_argument('--out-dir', type=Path, default=Path('out'))
    args = parser.parse_args()
    main(Args(**vars(args)))