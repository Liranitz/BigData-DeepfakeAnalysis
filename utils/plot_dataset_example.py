# save as: tools/preview_groups_grid.py
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import random

from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def load_image_safe(p: Path, size: int = 256) -> Image.Image:
    try:
        img = Image.open(p).convert("RGB")
    except (UnidentifiedImageError, OSError):
        # fallback placeholder
        img = Image.new("RGB", (size, size), color=(240, 240, 240))
    # simple thumbnail keeping aspect ratio
    img.thumbnail((size, size))
    return img

def find_genimage_groups(root: Path) -> List[Tuple[str, Path]]:
    """
    GenImage layout: root/imagenet_..._<MODEL>/train/ai/*.*
    Group name is the suffix after the last underscore.
    """
    out = []
    for d in sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith("imagenet")]):
        name = d.name.split("_")[-1] if "_" in d.name else d.name
        ai = d / "train" / "ai"
        if ai.is_dir():
            out.append((name, ai))
    return out

def find_forged_groups(root: Path) -> List[Tuple[str, Path]]:
    """
    Forged dataset layout: root/REAL and root/FAKE.
    """
    out = []
    for g in ["REAL", "FAKE"]:
        p = root / g
        if p.is_dir():
            out.append((g, p))
    return out

def sample_images(folder: Path, k: int) -> List[Path]:
    files = [p for p in folder.iterdir() if is_image(p)]
    files.sort()
    if not files:
        return []
    if k >= len(files):
        return files
    return random.sample(files, k)

def make_grid(
    group_to_images: Dict[str, List[Path]],
    cols: int,
    thumb: int,
    title_fontsize: int,
    out_path: Path,
) -> None:
    groups = list(group_to_images.keys())
    rows = (len(groups) + cols - 1) // cols
    if rows == 0:
        print("No groups to plot.")
        return

    # figure size heuristic
    w_in = cols * (thumb / 80)  # ~80 px per inch
    h_in = rows * (thumb / 80 + 0.4)  # extra for titles
    fig, axes = plt.subplots(rows, cols, figsize=(w_in, h_in))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, g in enumerate(groups):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        imgs = group_to_images[g]
        # show first sample per group
        img = load_image_safe(imgs[0], size=thumb)
        ax.imshow(img)
        # Map forged dataset labels
        title = g
        if g.upper() == "REAL":
            title = "AutoSplice-Real"
        elif g.upper() == "FAKE":
            title = "AutoSplice-Forged"
        ax.set_title(title, fontsize=title_fontsize, pad=6)
        ax.axis("off")

    # hide extra axes
    for idx in range(len(groups), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved grid -> {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Create a grid of example images per group with group name as title."
    )
    # New combined-mode args
    ap.add_argument("--genimage_root", type=str, default=None,
                    help="Root with imagenet_* dirs (GenImage layout).")
    ap.add_argument("--forged_root", type=str, default=None,
                    help="Root with REAL and FAKE subfolders.")
    # Backward compatible single source args
    ap.add_argument("--root", type=str, default=None,
                    help="Legacy: dataset root (use with --mode).")
    ap.add_argument("--mode", choices=["genimage", "forged"], default=None,
                    help="Legacy: folder layout type for --root.")
    # Display options
    ap.add_argument("--per_group", type=int, default=1,
                    help="Number of examples to sample per group (grid shows first one).")
    ap.add_argument("--max_groups", type=int, default=1000,
                    help="Cap number of groups to plot.")
    ap.add_argument("--cols", type=int, default=3, help="Grid columns (default 3).")
    ap.add_argument("--thumb", type=int, default=256, help="Thumbnail target size for display.")
    ap.add_argument("--title_font", type=int, default=12, help="Title fontsize.")
    ap.add_argument("--out", type=str, default="groups_preview.png", help="Output image path.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = ap.parse_args()

    random.seed(args.seed)

    # Collect groups from whichever roots were provided
    groups: List[Tuple[str, Path]] = []

    # Combined mode: both roots provided
    if args.genimage_root:
        gr = Path(args.genimage_root).resolve()
        if gr.exists():
            groups.extend(find_genimage_groups(gr))
        else:
            print(f"GenImage root not found: {gr}")

    if args.forged_root:
        fr = Path(args.forged_root).resolve()
        if fr.exists():
            groups.extend(find_forged_groups(fr))
        else:
            print(f"Forged root not found: {fr}")

    # Legacy single mode if no combined roots given
    if not groups and args.root and args.mode:
        root = Path(args.root).resolve()
        if not root.exists():
            print(f"Root not found: {root}")
            return
        if args.mode == "genimage":
            groups = find_genimage_groups(root)
        else:
            groups = find_forged_groups(root)

    if not groups:
        print("No groups found.")
        return

    # sample files
    group_to_images: Dict[str, List[Path]] = {}
    for gname, gpath in groups[: args.max_groups]:
        imgs = sample_images(gpath, k=max(1, args.per_group))
        if imgs:
            group_to_images[gname] = imgs

    if not group_to_images:
        print("No images found in any group.")
        return

    make_grid(
        group_to_images=group_to_images,
        cols=args.cols,
        thumb=args.thumb,
        title_fontsize=args.title_font,
        out_path=Path(args.out),
    )

if __name__ == "__main__":
    main()
