# src/extract_embeddings_by_imagenet_layout.py
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import timm
from timm.data import create_transform
try:
    from timm.data import resolve_model_data_config as _resolve_cfg
except Exception:
    from timm.data import resolve_data_config as _resolve_cfg  # fallback


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------
# Dataset utilities
# ----------------------------
def list_imagenet_projects(root: Path) -> List[Path]:
    """List all folders that start with 'imagenet'."""
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("imagenet")])


def parse_model_name(project_dir: Path) -> str:
    """Extract model name from imagenet folder name."""
    name = project_dir.name
    if "_" in name:
        return name.split("_")[-1]
    return name


class FlatImageSet(Dataset):
    """Simple flat dataset for reading all images in a directory."""
    def __init__(self, folder: Path, transform):
        self.folder = folder
        self.transform = transform
        self.items = [
            f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath = self.items[idx]
        try:
            img = Image.open(fpath).convert("RGB")
            key = fpath.name
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (224, 224))
            key = f"__SKIP__::{fpath.name}"
        return self.transform(img), key


# ----------------------------
# Model creation
# ----------------------------
def build_backbone(model_name: str, pretrained: bool, device: torch.device) -> tuple[nn.Module, any]:
    """
    Create a timm model that outputs pooled features.
    num_classes=0 → forward() returns [B, feat_dim].
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.to(device).eval()
    cfg = _resolve_cfg(model)
    tfm = create_transform(**cfg, is_training=False)
    return model, tfm


# ----------------------------
# Embedding extraction
# ----------------------------
@torch.no_grad()
def embed_folder(backbone: nn.Module, transform, folder: Path,
                 device: torch.device, batch_size: int, num_workers: int) -> Dict[str, list]:
    ds = FlatImageSet(folder, transform=transform)
    if len(ds) == 0:
        return {}

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    out: Dict[str, list] = {}
    for imgs, keys in tqdm(loader, desc=f"Embedding {folder}"):
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs)  # [B, feat_dim]
        feats = feats.float().cpu()
        for k, v in zip(keys, feats):
            if not k.startswith("__SKIP__::"):
                out[k] = v.tolist()
    return out


# ----------------------------
# Main logic
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    help="Root folder containing imagenet* projects (optional if using forged_data)")
    ap.add_argument("--forged_data", type=str,
                    help="Optional forged dataset with REAL and FAKE subfolders")
    ap.add_argument("--out_dir", default="outputs", type=str,
                    help="Output folder for JSON files")
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--model_name", default="xception", type=str,
                    help="Any timm model name (e.g., xception, resnet50, vit_base_patch16_224)")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use pretrained weights if available")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Create the chosen backbone
    backbone, tfm = build_backbone(args.model_name, args.pretrained, device)

    # ----------------------------------------------------
    # (A) Handle ImageNet-style GenImage datasets
    # ----------------------------------------------------
    if args.data_root:
        data_root = Path(args.data_root).resolve()
        projects = list_imagenet_projects(data_root)

        for proj in projects:
            model_suffix = parse_model_name(proj)
            ai_folder = proj / "train" / "ai"
            if not ai_folder.exists():
                print(f"Skipping {proj.name} - train/ai not found.")
                continue

            name_to_embedding = embed_folder(backbone, tfm, ai_folder,
                                             device, args.batch_size, args.num_workers)
            if not name_to_embedding:
                print(f"No images in {ai_folder} - skipping write.")
                continue

            json_path = out_dir / f"{proj.name}__{model_suffix}__{args.model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(name_to_embedding, f, ensure_ascii=False)
            print(f"Wrote {len(name_to_embedding)} embeddings -> {json_path}")

    # ----------------------------------------------------
    # (B) Handle Forged dataset (REAL / FAKE subfolders)
    # ----------------------------------------------------
    if args.forged_data:
        forged_root = Path(args.forged_data).resolve()
        merged: Dict[str, list] = {}

        for label in ["REAL", "FAKE"]:
            subdir = forged_root / label
            if not subdir.exists():
                print(f"Skipping missing {label} folder at {subdir}")
                continue

            name_to_embedding = embed_folder(backbone, tfm, subdir,
                                             device, args.batch_size, args.num_workers)
            if not name_to_embedding:
                print(f"No images in {subdir} - skipping write.")
                continue

            # Save individual REAL / FAKE files
            json_path = out_dir / f"{label}__{args.model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(name_to_embedding, f, ensure_ascii=False)
            print(f"Wrote {len(name_to_embedding)} embeddings -> {json_path}")

            # Also merge them for a combined analysis later
            merged.update({f"{label}/{k}": v for k, v in name_to_embedding.items()})

        # Save a merged combined file
        if merged:
            merged_path = out_dir / f"forged_dataset__REAL_FAKE__{args.model_name}.json"
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False)
            print(f"\n✅ Combined REAL+FAKE embeddings -> {merged_path}")

    print("\n✅ Done. All embeddings exported to:", out_dir)


if __name__ == "__main__":
    main()
