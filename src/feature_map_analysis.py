# src/feature_map_analysis_vit_ready.py
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import timm
from timm.data import resolve_data_config, create_transform
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------------
# Utilities
# ----------------------------
def find_imagenet_projects(root: Path) -> List[Tuple[str, Path]]:
    out = []
    for p in sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith("imagenet")]):
        model_name = p.name.split("_")[-1] if "_" in p.name else p.name
        ai_folder = p / "train" / "ai"
        if ai_folder.is_dir():
            out.append((model_name, ai_folder))
    return out

def list_images(folder: Path) -> List[Path]:
    return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])

def list_first_n_images(folder: Path, n: int) -> List[Path]:
    files = list_images(folder)
    return files[: min(n, len(files))]

def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def select_channels(x: torch.Tensor, k: int = 8) -> List[int]:
    # x shape: [C, H, W]
    c = x.shape[0]
    if k >= c:
        return list(range(c))
    var = x.view(c, -1).var(dim=1)
    return torch.topk(var, k=k).indices.cpu().tolist()

def save_feature_grid(fmap: torch.Tensor, out_png: Path, title: str, max_channels: int = 8) -> None:
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
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        ax.imshow(m.numpy(), cmap="gray")
        ax.set_title(f"ch {ci}", fontsize=9)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

def save_original(img_t: torch.Tensor, out_png: Path) -> None:
    x = img_t.detach().cpu().clone()
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
    out_indices: Tuple[int, ...]
    batch_size: int = 16

class FeatureExtractor:
    """
    Works for CNNs and ViTs through timm features_only.
    For ViTs, features_only exposes spatial feature maps after patch embedding
    and selected transformer blocks. Shapes are still [B, C, H, W].
    """
    def __init__(self, spec: BackboneSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.model = timm.create_model(
            spec.timm_id, pretrained=True, features_only=True, out_indices=spec.out_indices
        ).to(device).eval()
        self.cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.cfg)

    @torch.no_grad()
    def extract(self, img_pil: Image.Image):
        x = self.transform(img_pil).unsqueeze(0).to(self.device)
        feats = self.model(x)  # list of [1, C, H, W]
        feats = [f.squeeze(0).cpu() for f in feats]
        return x.squeeze(0).cpu(), feats

    def layer_names(self) -> List[str]:
        return [f"stage{idx}" for idx in self.spec.out_indices]

# ----------------------------
# Registry of supported models
# Add new models here and they become available via --models
# ----------------------------
SPECS: Dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(name="resnet50", timm_id="resnet50", out_indices=(1, 2, 3)),
    "xception": BackboneSpec(name="xception", timm_id="xception", out_indices=(1, 2, 3)),
    # ViT base - pick mid to deep blocks. For vit_base_patch16_224, valid out_indices include 1..11.
    "vit_base_patch16_224": BackboneSpec(name="vit_base_patch16_224", timm_id="vit_base_patch16_224", out_indices=(1, 5, 9)),
    # You can easily add more ViTs, for example:
    # "vit_small_patch16_224": BackboneSpec(name="vit_small_patch16_224", timm_id="vit_small_patch16_224", out_indices=(4, 7, 11)),
}

def build_extractors(requested: Optional[List[str]], device: torch.device) -> Dict[str, FeatureExtractor]:
    """
    requested None -> build all supported
    requested list -> build only those (case insensitive), error if unknown
    """
    if requested is None or len(requested) == 0:
        names = sorted(SPECS.keys())
    else:
        names = []
        for r in requested:
            key = r.lower()
            if key not in SPECS:
                raise ValueError(f"unsupported model '{r}'. supported: {sorted(SPECS.keys())}")
            names.append(key)
    return {name: FeatureExtractor(SPECS[name], device) for name in names}

# ----------------------------
# Forged dataset helper
# ----------------------------
def get_forged_sets(forged_root: Path, per_class_count: int = 100) -> List[Tuple[str, List[Path]]]:
    subdirs = {p.name.lower(): p for p in forged_root.iterdir() if p.is_dir()}
    fake_dir = subdirs.get("fake")
    real_dir = subdirs.get("real")
    groups: List[Tuple[str, List[Path]]] = []
    if fake_dir and fake_dir.is_dir():
        groups.append(("FAKE", list_first_n_images(fake_dir, per_class_count)))
    if real_dir and real_dir.is_dir():
        groups.append(("REAL", list_first_n_images(real_dir, per_class_count)))
    return groups

# ----------------------------
# Main analysis loop
# ----------------------------
def analyze(
    data_root: Optional[Path],
    out_root: Path,
    per_model_samples: int,
    models_to_run: Optional[List[str]],
    seed: int,
    device_str: str,
    max_channels_plot: int,
    forged_data: Optional[Path] = None
) -> None:
    device = torch.device(device_str)
    extractors = build_extractors(models_to_run, device)
    rng = np.random.default_rng(seed)

    # Option A: REAL/FAKE dataset
    if forged_data is not None:
        if not forged_data.exists() or not forged_data.is_dir():
            print(f"--forged_data path does not exist or is not a directory: {forged_data}")
            return
        groups = get_forged_sets(forged_data, per_class_count=100)
        if not groups:
            print(f"No FAKE or REAL subfolders found under {forged_data}")
            return

        for group_name, sample_paths in groups:
            if not sample_paths:
                print(f"Skip {group_name} - no images")
                continue
            for arch_name, extractor in extractors.items():
                print(f"[{group_name}] {arch_name} - processing {len(sample_paths)} images")
                for img_path in tqdm(sample_paths):
                    try:
                        pil = load_rgb(img_path)
                    except Exception:
                        continue
                    x_t, feats = extractor.extract(pil)
                    layer_names = extractor.layer_names()
                    img_stem = img_path.stem
                    base_dir = out_root / arch_name / group_name / img_stem
                    base_dir.mkdir(parents=True, exist_ok=True)
                    save_original(x_t, base_dir / "input_transformed.png")
                    for lname, fmap in zip(layer_names, feats):
                        save_feature_grid(
                            fmap, base_dir / f"{lname}_grid.png",
                            title=f"{arch_name} - {group_name} - {lname}",
                            max_channels=max_channels_plot
                        )
                        np.savez_compressed(base_dir / f"{lname}.npz", fmap=fmap.numpy())
        print(f"Done. Outputs saved under {out_root}")
        return

    # Option B: GenImage layout
    if data_root is None:
        print("either --forged_data must be provided or --data_root must be set")
        return
    if not data_root.exists() or not data_root.is_dir():
        print(f"--data_root path does not exist or is not a directory: {data_root}")
        return

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

        for arch_name, extractor in extractors.items():
            print(f"[{gen_name}] {arch_name} - processing {len(sample_paths)} images")
            for img_path in tqdm(sample_paths):
                try:
                    pil = load_rgb(img_path)
                except Exception:
                    continue
                x_t, feats = extractor.extract(pil)
                layer_names = extractor.layer_names()
                img_stem = img_path.stem
                base_dir = out_root / arch_name / gen_name / img_stem
                base_dir.mkdir(parents=True, exist_ok=True)
                save_original(x_t, base_dir / "input_transformed.png")
                for lname, fmap in zip(layer_names, feats):
                    save_feature_grid(
                        fmap, base_dir / f"{lname}_grid.png",
                        title=f"{arch_name} - {gen_name} - {lname}",
                        max_channels=max_channels_plot
                    )
                    np.savez_compressed(base_dir / f"{lname}.npz", fmap=fmap.numpy())
    print(f"Done. Outputs saved under {out_root}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=False,
                    help="Root with imagenet*/train/ai. Optional if --forged_data is set")
    ap.add_argument("--out_dir", default="out", type=str, help="Output root directory")
    ap.add_argument("--per_model_samples", default=100, type=int,
                    help="How many images to sample per generator model")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of models to load. If omitted, load all supported. Examples: --models resnet50 xception vit_base_patch16_224")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--max_channels_plot", default=8, type=int,
                    help="How many channels to show in each grid preview")
    ap.add_argument("--forged_data", default=None, type=str,
                    help="Optional path with FAKE and REAL subfolders. If set, take first 100 from each")
    args = ap.parse_args()

    analyze(
        data_root=(Path(args.data_root).resolve() if args.data_root else None),
        out_root=Path(args.out_dir).resolve(),
        per_model_samples=int(args.per_model_samples),
        models_to_run=(args.models if args.models else None),
        seed=int(args.seed),
        device_str=args.device,
        max_channels_plot=int(args.max_channels_plot),
        forged_data=(Path(args.forged_data).resolve() if args.forged_data else None)
    )

if __name__ == "__main__":
    main()
