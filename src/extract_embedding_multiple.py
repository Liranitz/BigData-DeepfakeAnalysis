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
    # timm >= 0.9 API
    from timm.data import resolve_model_data_config as _resolve_cfg
except Exception:
    # older timm fallback
    from timm.data import resolve_data_config as _resolve_cfg  # type: ignore

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_imagenet_projects(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("imagenet")])

def parse_model_name(project_dir: Path) -> str:
    name = project_dir.name
    if "_" in name:
        return name.split("_")[-1]
    return name

class FlatImageSet(Dataset):
    def __init__(self, folder: Path, transform):
        self.folder = folder
        self.transform = transform
        self.items = []
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                self.items.append(f)

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

def build_backbone(model_name: str, pretrained: bool, device: torch.device) -> tuple[nn.Module, any]:
    """
    Create a timm model with pooled feature output.
    num_classes=0 makes forward() return pooled features [B, feat_dim].
    Returns the model and its eval transform.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.to(device).eval()

    cfg = _resolve_cfg(model)
    tfm = create_transform(**cfg, is_training=False)
    return model, tfm

@torch.no_grad()
def embed_folder(
    backbone: nn.Module,
    transform,
    folder: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int
) -> Dict[str, list]:
    ds = FlatImageSet(folder, transform=transform)
    if len(ds) == 0:
        return {}

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    out: Dict[str, list] = {}
    for imgs, keys in tqdm(loader, desc=f"Embedding {folder}"):
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs)              # shape [B, feat_dim] because num_classes=0
        feats = feats.float().cpu()
        for k, v in zip(keys, feats):
            if k.startswith("__SKIP__::"):
                continue
            out[k] = v.tolist()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str,
                    help="Folder that contains imagenet* projects")
    ap.add_argument("--out_dir", default="outputs", type=str)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    # new dynamic model args
    ap.add_argument("--model_name", default="xception", type=str,
                    help="Any timm model name, e.g. xception, xception41, resnet50, vit_base_patch16_224")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use pretrained weights if available")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # build chosen backbone and its correct eval transform
    backbone, tfm = build_backbone(args.model_name, args.pretrained, device)

    projects = list_imagenet_projects(data_root)
    if not projects:
        print("No imagenet* projects found under data_root.")
        return

    for proj in projects:
        model_suffix = parse_model_name(proj)  # kept for file naming compatibility
        ai_folder = proj / "train" / "ai"

        if not ai_folder.exists() or not ai_folder.is_dir():
            print(f"Skipping {proj.name} - train/ai not found.")
            continue

        name_to_embedding = embed_folder(
            backbone=backbone,
            transform=tfm,
            folder=ai_folder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        if not name_to_embedding:
            print(f"No images in {ai_folder} - skipping write.")
            continue

        # Include actual backbone name in filename so you know what produced it
        json_path = out_dir / f"{proj.name}__{model_suffix}__{args.model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(name_to_embedding, f, ensure_ascii=False)
        print(f"Wrote {len(name_to_embedding)} embeddings -> {json_path}")

if __name__ == "__main__":
    main()
