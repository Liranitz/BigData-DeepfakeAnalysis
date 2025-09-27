import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Use timm for unified feature extraction across models
import timm
from timm.data import resolve_data_config, create_transform

import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------
# Utilities
# ----------------------------

def find_imagenet_projects(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns list of (model_name, ai_folder_path) for each imagenet* project.
    We parse model name as text after the last '_' in dir name.
    """
    out = []
    for p in sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith("imagenet")]):
        model_name = p.name.split("_")[-1] if "_" in p.name else p.name
        ai_folder = p / "train" / "ai"
        if ai_folder.is_dir():
            out.append((model_name, ai_folder))
    return out


def list_images(folder: Path) -> List[Path]:
    return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def select_channels(x: torch.Tensor, k: int = 8) -> List[int]:
    """
    Given a feature map [C, H, W], pick k channels with the highest spatial variance.
    Returns channel indices.
    """
    c = x.shape[0]
    if k >= c:
        return list(range(c))
    # variance per channel
    var = x.view(c, -1).var(dim=1)
    topk = torch.topk(var, k=k).indices.cpu().tolist()
    return topk


def save_feature_grid(
    fmap: torch.Tensor,
    out_png: Path,
    title: str,
    max_channels: int = 8
) -> None:
    """
    fmap: [C, H, W] tensor (cpu)
    Save a grid of up to max_channels channels.
    """
    fmap = fmap.detach().cpu()
    ch_idx = select_channels(fmap, k=max_channels)
    k = len(ch_idx)

    cols = min(4, k)
    rows = int(np.ceil(k / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    plt.suptitle(title, fontsize=12, y=0.98)
    for i, ci in enumerate(ch_idx, 1):
        ax = plt.subplot(rows, cols, i)
        m = fmap[ci]
        # normalize per channel for visibility
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        ax.imshow(m.numpy(), cmap="gray")
        ax.set_title(f"ch {ci}", fontsize=9)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_original(img_t: torch.Tensor, out_png: Path) -> None:
    """
    img_t is a tensor after transform, shape [C, H, W], normalized.
    We will unnormalize using imagenet stats if present in filename hint,
    or just rescale to [0,1] safely.
    """
    x = img_t.detach().cpu().clone()
    # Best effort unnorm for ImageNet defaults
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(x).save(out_png)


# ----------------------------
# Modular backbone interface
# ----------------------------

@dataclass
class BackboneSpec:
    name: str
    timm_id: str
    out_indices: Tuple[int, ...]  # which stages to return
    batch_size: int = 16


class FeatureExtractor:
    """
    Wraps a timm backbone with features_only=True. Provides:
    - transform
    - extract(img) -> List[torch.Tensor] with feature maps at out_indices
    """
    def __init__(self, spec: BackboneSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.model = timm.create_model(
            spec.timm_id,
            pretrained=True,
            features_only=True,
            out_indices=spec.out_indices
        ).to(device).eval()

        self.cfg = resolve_data_config({}, model=self.model)
        # Force ImageNet like size 224 for resnet50 and xception default
        self.transform = create_transform(**self.cfg)

    @torch.no_grad()
    def extract(self, img_pil: Image.Image) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns the input tensor after transform and a list of feature maps as cpu tensors.
        """
        x = self.transform(img_pil).unsqueeze(0).to(self.device)  # [1, C, H, W]
        feats: List[torch.Tensor] = self.model(x)                 # list of [1, C, H, W]
        feats = [f.squeeze(0).cpu() for f in feats]
        return x.squeeze(0).cpu(), feats

    def layer_names(self) -> List[str]:
        # Make neat names per index
        return [f"stage{idx}" for idx in self.spec.out_indices]


# ----------------------------
# Registry for models
# ----------------------------

def build_registry(device: torch.device, bs_override: int | None = None) -> Dict[str, FeatureExtractor]:
    """
    Returns a dict of {arch_name: FeatureExtractor}.
    - ResNet-50 via timm id 'resnet50'
    - Xception via timm id 'xception'
    We pick 3 higher stages for a good look at semantics.
    """
    registry: Dict[str, FeatureExtractor] = {}

    resnet_spec = BackboneSpec(
        name="resnet50",
        timm_id="resnet50",
        out_indices=(2, 3, 4),  # C3, C4, C5 like stages
        batch_size=bs_override or 16
    )
    xception_spec = BackboneSpec(
        name="xception",
        timm_id="xception",
        out_indices=(1, 2, 3),  # 3 deeper stages in timm xception
        batch_size=bs_override or 16
    )

    registry["resnet50"] = FeatureExtractor(resnet_spec, device)
    registry["xception"] = FeatureExtractor(xception_spec, device)
    return registry


# ----------------------------
# Main analysis loop
# ----------------------------

def analyze(
    data_root: Path,
    out_root: Path,
    per_model_samples: int,
    models_to_run: List[str],
    seed: int,
    device_str: str,
    max_channels_plot: int
) -> None:
    device = torch.device(device_str)
    registry = build_registry(device)
    for m in list(registry.keys()):
        if m not in models_to_run:
            registry.pop(m)

    rng = np.random.default_rng(seed)

    projects = find_imagenet_projects(data_root)
    if not projects:
        print(f"No imagenet* projects found under {data_root}")
        return

    for gen_name, ai_folder in projects:
        images = list_images(ai_folder)
        if not images:
            print(f"Skip {gen_name} - no images in {ai_folder}")
            continue

        if len(images) > per_model_samples:
            idx = rng.choice(len(images), size=per_model_samples, replace=False)
            sample_paths = [images[i] for i in idx]
        else:
            sample_paths = images

        for arch_name, extractor in registry.items():
            print(f"[{gen_name}] {arch_name} - processing {len(sample_paths)} images")
            for img_path in tqdm(sample_paths):
                try:
                    pil = load_rgb(img_path)
                except Exception:
                    continue

                x_t, feats = extractor.extract(pil)
                layer_names = extractor.layer_names()

                # out path: out/<arch>/<generator>/<img_stem>/
                img_stem = img_path.stem
                base_dir = out_root / arch_name / gen_name / img_stem
                base_dir.mkdir(parents=True, exist_ok=True)

                # save original (transformed) for reference
                save_original(x_t, base_dir / "input_transformed.png")

                # save each layer grid and raw tensor
                for lname, fmap in zip(layer_names, feats):
                    # grid preview
                    save_feature_grid(
                        fmap=fmap,
                        out_png=base_dir / f"{lname}_grid.png",
                        title=f"{arch_name} - {gen_name} - {lname}",
                        max_channels=max_channels_plot
                    )
                    # raw as npz
                    np.savez_compressed(base_dir / f"{lname}.npz", fmap=fmap.numpy())

    print(f"Done. Outputs saved under {out_root}")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str,
                    help="Root folder that contains imagenet*/train/ai")
    ap.add_argument("--out_dir", default="out", type=str,
                    help="Output root directory")
    ap.add_argument("--per_model_samples", default=100, type=int,
                    help="How many images to sample per generator model")
    ap.add_argument("--models", nargs="+", default=["resnet50", "xception"],
                    help="Which backbones to run. Example: --models resnet50 xception")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--max_channels_plot", default=8, type=int,
                    help="How many channels to show in each grid preview")
    args = ap.parse_args()

    analyze(
        data_root=Path(args.data_root).resolve(),
        out_root=Path(args.out_dir).resolve(),
        per_model_samples=int(args.per_model_samples),
        models_to_run=[m.lower() for m in args.models],
        seed=int(args.seed),
        device_str=args.device,
        max_channels_plot=int(args.max_channels_plot)
    )


if __name__ == "__main__":
    main()
