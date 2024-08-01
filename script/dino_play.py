import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch import ByteTensor, FloatTensor, inference_mode
from torch.amp import autocast
# from torch.nested import nested_tensor
from torchvision.io.image import ImageReadMode, read_image, decode_jpeg, read_file
from torchvision.transforms.v2 import Resize, CenterCrop, Compose, Normalize, InterpolationMode
from dino_embed.chunk import chunk
from typing import Any, Iterable, Protocol
from functools import partial
import piexif
from multiprocessing.pool import ThreadPool, MapResult

@dataclass
class Args:
    in_dir: Path
    out_dir: Path
    batch_size: int

class TransformFactory(Protocol):
    @staticmethod
    def __call__(shortest_side: int) -> Compose: ...

jpeg_suffices = {'.jpg', '.jpeg'}
def load_image(p: Path, transform_factory: TransformFactory, device=torch.device('cpu')) -> ByteTensor:
    nvjpeg_eligible = p.suffix in jpeg_suffices and device.type == 'cuda'
    if nvjpeg_eligible:
        exif_dict: dict[str, Any] = piexif.load(str(p))
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
            # https://pillow.readthedocs.io/en/latest/_modules/PIL/ImageOps.html#exif_transpose
            wants_reorient = orientation > 1 and orientation < 9
            # I mean we could implement some tensor-rotating and flipping, but they're rare enough we should avoid such complexity
            nvjpeg_eligible &= not wants_reorient

    if nvjpeg_eligible:
        data: ByteTensor = read_file(str(p))
        # NOTE: we deliberately forego apply_exif_orientation, because nvjpeg doesn't support it.
        #       instead we will, uh, hope none of these images rely on EXIF for orientation.
        img: ByteTensor = decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
    else:
        img: ByteTensor = read_image(str(p), mode=ImageReadMode.RGB, apply_exif_orientation=True).to(device)
    tform: Compose = transform_factory(min(img.shape[-2:]))
    tformed: FloatTensor = tform(img.half())#.float()
    return tformed.unsqueeze(0)

img_suffices = {*jpeg_suffices, '.png'}
def main(args: Args) -> None:
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # we could make this lazier, but I'd rather have a progress bar
    candidates: list[Path] = [p for p in args.in_dir.rglob('**/*') if p.suffix in img_suffices]

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').eval().to(device)
    patch_size: int = dinov2.patch_size

    def transform_factory(shortest_side: int) -> Compose:
        nearest_multiple_px: int = (shortest_side//patch_size)*patch_size
        clamped_px = min(nearest_multiple_px, 448)
        # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) # assumes input is 0 to 1
        # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L55
        # https://github.com/facebookresearch/dinov2/issues/2
        return Compose([
            Resize(clamped_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(clamped_px),
            Normalize(
                mean=(123.675, 116.28, 103.53), # assumes input is 0 to 255
                std=(58.395, 57.12, 57.375),
                inplace=True,
            ),
        ])

    batch_iter: Iterable[tuple[Path, ...]] = chunk(candidates, args.batch_size)
    with ThreadPool(processes=args.batch_size) as pool:
        batch_paths: tuple[Path, ...] = next(batch_iter)
        batch_co: MapResult[ByteTensor] = pool.map_async(partial(load_image, device=device, transform_factory=transform_factory), batch_paths)
        for batch_paths in tqdm(batch_iter):
            batch: list[ByteTensor] = batch_co.get()
            batch_co: MapResult[ByteTensor] = pool.map_async(partial(load_image, device=device, transform_factory=transform_factory), batch_paths)
            # batch_t = nested_tensor(batch)
            with inference_mode(), autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                batch_out: list[dict[str, FloatTensor]] = dinov2.forward_features_list(batch, masks_list=[None]*len(batch))
                batch_emb: list[FloatTensor] = [e['x_norm_clstoken'] for e in batch_out]
                emb_stack = torch.cat(batch_emb, dim=0)
                assert not emb_stack.isnan().any().item()
                pass
            pass
        batch: list[ByteTensor] = batch_co.get()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dino Play')
    parser.add_argument('--in-dir', type=Path, default=Path('in'))
    parser.add_argument('--out-dir', type=Path, default=Path('out'))
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    main(Args(**vars(args)))