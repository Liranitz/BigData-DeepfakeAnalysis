# src/extract_embeddings_by_imagenet_layout.py
import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_imagenet_projects(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("imagenet")])

def parse_model_name(project_dir: Path) -> str:
    # Take substring after the last underscore
    name = project_dir.name
    if "_" in name:
        return name.split("_")[-1]
    return name  # fallback

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
            key = fpath.name  # just the file name as requested
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (224, 224))
            key = f"__SKIP__::{fpath.name}"
        return self.transform(img), key

def build_resnet50(device: torch.device) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*(list(m.children())[:-1]))  # up to avgpool
    backbone.to(device).eval()
    return backbone

@torch.no_grad()
def embed_folder(
    backbone: nn.Module,
    folder: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int
) -> Dict[str, list]:
    # Preprocessing for ImageNet-trained ResNet-50
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build dataset (loads images and applies tfm)
    ds = FlatImageSet(folder, transform=tfm)
    if len(ds) == 0:
        return {}

    # DataLoader
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    out = {}
    for imgs, keys in tqdm(loader, desc=f"Embedding {folder}"):
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs).squeeze(-1).squeeze(-1).cpu()  # [B, 2048]
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
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    backbone = build_resnet50(device)

    projects = list_imagenet_projects(data_root)
    if not projects:
        print("No imagenet* projects found under data_root.")
        return

    for proj in projects:
        model_name = parse_model_name(proj)
        ai_folder = proj / "train" / "ai"

        if not ai_folder.exists() or not ai_folder.is_dir():
            print(f"Skipping {proj.name} - train/ai not found.")
            continue

        name_to_embedding = embed_folder(
            backbone=backbone, folder=ai_folder, device=device,
            batch_size=args.batch_size, num_workers=args.num_workers
        )

        if not name_to_embedding:
            print(f"No images in {ai_folder} - skipping write.")
            continue

        # One JSON per project
        json_path = out_dir / f"{proj.name}__{model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(name_to_embedding, f, ensure_ascii=False)
        print(f"Wrote {len(name_to_embedding)} embeddings -> {json_path}")

if __name__ == "__main__":
    main()
