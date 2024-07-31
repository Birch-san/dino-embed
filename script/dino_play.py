import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from dino_embed.chunk import chunk

@dataclass
class Args:
    in_dir: Path
    out_dir: Path

img_suffices = {'.jpg', '.jpeg', '.png'}
def main(args: Args) -> None:
    # we could make this lazier, but I'd rather have a progress bar
    candidates: list[Path] = [p for p in args.in_dir.rglob('**/*') if p.suffix in img_suffices]
    for p in tqdm(candidates):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dino Play')
    parser.add_argument('--in-dir', type=Path, default=Path('in'))
    parser.add_argument('--out-dir', type=Path, default=Path('out'))
    args = parser.parse_args()
    main(Args(**vars(args)))