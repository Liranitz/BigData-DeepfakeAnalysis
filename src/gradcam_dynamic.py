import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import timm
from timm.data import resolve_data_config, create_transform

import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------------
# Utils
# ----------------------------
def list_images_in_folder(folder: Path, max_images: Optional[int] = None) -> List[Path]:
    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
    if max_images is not None and len(files) > max_images:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(files), size=max_images, replace=False)
        files = [files[i] for i in idx]
    return files

def list_imagenet_projects(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns (group_name, ai_folder) for each imagenet* project.
    group_name parsed from folder name after last '_', fallback to full name.
    """
    out = []
    for p in sorted([x for x in root.iterdir() if x.is_dir() and x.name.startswith("imagenet")]):
        group = p.name.split("_")[-1] if "_" in p.name else p.name
        ai = p / "train" / "ai"
        if ai.is_dir():
            out.append((group, ai))
    return out

def collect_image_paths(
    data_root: Optional[Path],
    forged_root: Optional[Path],
    files: Optional[List[Path]],
    per_group: int
) -> Dict[str, List[Path]]:
    """
    Returns dict group -> list[Path]
    Groups from:
      - imagenet*/train/ai
      - forged REAL/FAKE
      - explicit files (group='custom')
    """
    groups: Dict[str, List[Path]] = {}

    if data_root and data_root.exists():
        for group, ai in list_imagenet_projects(data_root):
            groups[group] = list_images_in_folder(ai, max_images=per_group)

    if forged_root and forged_root.exists():
        for lab in ["REAL", "FAKE"]:
            d = forged_root / lab
            if d.is_dir():
                groups[lab] = list_images_in_folder(d, max_images=per_group)

    if files:
        # treat as a single group unless you want to split by parent name
        groups.setdefault("custom", [])
        for fp in files:
            if fp.exists() and fp.suffix.lower() in IMG_EXTS:
                groups["custom"].append(fp)

        if per_group and len(groups["custom"]) > per_group:
            groups["custom"] = groups["custom"][:per_group]

    # prune empties
    groups = {k: v for k, v in groups.items() if v}
    return groups

def load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def to_numpy_img_from_tensor(x: torch.Tensor, mean=None, std=None) -> np.ndarray:
    # x is [C,H,W], possibly normalized
    y = x.detach().cpu().clone()
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        y = y * std + mean
    y = y.clamp(0,1)
    y = (y.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return y

def save_overlay(base_rgb: np.ndarray, heat: np.ndarray, out_path: Path, alpha: float = 0.35):
    # base_rgb uint8 [H,W,3], heat float [H,W] in [0,1]
    plt.figure(figsize=(5,5))
    plt.imshow(base_rgb)
    plt.imshow(heat, cmap="jet", alpha=alpha, interpolation="bilinear")
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_heatmap(heat: np.ndarray, out_path: Path):
    plt.figure(figsize=(5,5))
    plt.imshow(heat, cmap="jet")
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()

# ----------------------------
# Model builders
# ----------------------------
def build_model_and_transform(model_name: str, device: torch.device, pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained)
    model.to(device).eval()
    cfg = resolve_data_config({}, model=model)
    tfm = create_transform(**cfg)
    imsize = cfg.get("input_size", (3, 224, 224))[1:]
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    is_vit = "vit" in model_name.lower() or "deit" in model_name.lower() or "swin" in model_name.lower()
    return model, tfm, imsize, mean, std, is_vit

# ----------------------------
# Grad-CAM for CNNs
# ----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_h = target_layer.register_forward_hook(fwd_hook)
        self.bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    def remove(self):
        self.fwd_h.remove()
        self.bwd_h.remove()

    def __call__(self, model_inputs: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        """
        model_inputs: [1,3,H,W]
        returns heatmap [H,W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(model_inputs)  # [1, num_classes] or features
        if logits.ndim == 2 and logits.shape[1] > 1:
            if target_index is None:
                target_index = int(logits.argmax(dim=1).item())
            loss = logits[:, target_index].sum()
        else:
            # if model is num_classes=0 features, just take L2 norm as a proxy target
            loss = logits.norm(p=2)
        loss.backward()

        A = self.activations  # [B,C,h,w]
        G = self.gradients    # [B,C,h,w]
        weights = G.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=True)  # [B,1,h,w]
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)  # [h,w]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # upsample to input size
        cam_up = F.interpolate(cam[None,None], size=model_inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze().cpu().numpy()
        return cam_up

def auto_find_last_conv(model: nn.Module) -> nn.Module:
    """
    Heuristic: returns the last nn.Conv2d module found in a depth-first walk.
    Works well for ResNet and Xception in timm.
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last

def resolve_target_layer(model: nn.Module, layer_name: Optional[str]) -> nn.Module:
    if layer_name:
        # resolve dotted path, e.g., 'layer4.2.conv3'
        obj = model
        for part in layer_name.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        if not isinstance(obj, nn.Module):
            raise RuntimeError(f"Target layer {layer_name} not found as nn.Module")
        return obj
    return auto_find_last_conv(model)

# ----------------------------
# ViT Attention Rollout
# ----------------------------
class ViTRollout:
    """
    Simple attention rollout that multiplies attention matrices across blocks and
    extracts contribution to the CLS token. Produces a spatial map over patches.
    """
    def __init__(self, model: nn.Module, imsize: Tuple[int,int], patch_size: Optional[int] = None):
        self.model = model
        self.blocks = []
        # gather attention tensors via hooks
        self.attns: List[torch.Tensor] = []
        self.hooks = []

        # find transformer blocks (timm vit: model.blocks is common)
        if hasattr(model, "blocks"):
            self.blocks = list(model.blocks)
        else:
            # try to find any sequential named 'blocks'
            for name, m in model.named_modules():
                if name.endswith("blocks") and isinstance(m, nn.ModuleList):
                    self.blocks = list(m)
                    break
        if not self.blocks:
            raise RuntimeError("Could not locate ViT blocks for attention rollout")

        for blk in self.blocks:
            # try to hook the softmax attention output
            if hasattr(blk, "attn") and hasattr(blk.attn, "attn_drop"):
                def make_hook():
                    def hook(module, inp, out):
                        # inp[0] is attn probs before dropout in many timm impls
                        # however, attn_drop is Dropout, so 'inp' is softmax(attn)
                        self.attns.append(inp[0].detach())
                    return hook
                self.hooks.append(blk.attn.attn_drop.register_forward_hook(make_hook()))
            elif hasattr(blk, "attn"):
                # fallback: register hook on attn module output directly
                def make_hook2():
                    def hook(module, inp, out):
                        # hope 'out' are attention probs
                        self.attns.append(out.detach())
                    return hook
                self.hooks.append(blk.attn.register_forward_hook(make_hook2()))
            else:
                raise RuntimeError("ViT block without 'attn' attribute")

        # infer patch size from model if possible
        self.patch_size = patch_size
        if self.patch_size is None:
            # try timm convention
            if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
                ps = model.patch_embed.patch_size
                self.patch_size = ps if isinstance(ps, int) else ps[0]
            else:
                self.patch_size = 16  # default guess

        self.im_h, self.im_w = imsize

    def remove(self):
        for h in self.hooks:
            h.remove()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        x: [1,3,H,W]
        Returns a heatmap [H,W] in [0,1]
        """
        self.attns.clear()
        _ = self.model(x)  # forward pass collects attentions

        if not self.attns:
            raise RuntimeError("No attention maps captured from ViT")

        # Each attention: [B, heads, tokens, tokens]
        # Rollout: multiply (I + A) across layers, average heads
        attn_avg = []
        for a in self.attns:
            # a may be dropped out probabilities, clamp
            a = a.float().mean(dim=1)  # average over heads -> [B, T, T]
            # add identity for residual attention
            eye = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
            a = a + eye
            # normalize rows
            a = a / a.sum(dim=-1, keepdim=True)
            attn_avg.append(a)

        rollout = attn_avg[0]
        for a in attn_avg[1:]:
            rollout = rollout @ a

        rollout = rollout[0]  # [T, T]
        # CLS token index is 0 by convention in ViT
        cls_attn = rollout[0, 1:]  # drop CLS to patches -> [num_patches]
        num_patches = cls_attn.numel()
        gh = self.im_h // self.patch_size
        gw = self.im_w // self.patch_size
        if gh * gw != num_patches:
            # try to guess square
            L = int(np.sqrt(num_patches))
            gh, gw = L, L
        cam = cls_attn.reshape(gh, gw)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_up = F.interpolate(cam[None,None], size=(self.im_h, self.im_w), mode="bilinear", align_corners=False)
        return cam_up.squeeze().cpu().numpy()

# ----------------------------
# Main loop
# ----------------------------
def process_images_for_model(
    model_name: str,
    groups: Dict[str, List[Path]],
    out_root: Path,
    device: torch.device,
    target_layer_str: Optional[str] = None,
    pretrained: bool = True
):
    model, tfm, imsize, mean, std, is_vit = build_model_and_transform(model_name, device, pretrained)

    if is_vit:
        rollout = ViTRollout(model, imsize=imsize)
        cam_fn = None
    else:
        layer = resolve_target_layer(model, target_layer_str)
        cam = GradCAM(model, layer)
        rollout = None
        cam_fn = cam

    for group, paths in groups.items():
        for img_path in paths:
            try:
                pil = load_pil(img_path)
            except Exception:
                continue
            x = tfm(pil).unsqueeze(0).to(device)
            base_rgb = to_numpy_img_from_tensor(x.squeeze(0), mean=mean, std=std)

            if is_vit:
                heat = rollout(x)
            else:
                heat = cam_fn(x)

            # save
            stem = img_path.stem
            base_dir = out_root / model_name / group
            save_heatmap(heat, base_dir / f"{stem}_cam.png")
            save_overlay(base_rgb, heat, base_dir / f"{stem}_overlay.png", alpha=0.35)

    if is_vit and rollout:
        rollout.remove()
    if not is_vit and cam_fn:
        cam_fn.remove()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, help="Root with imagenet*/train/ai folders")
    ap.add_argument("--forged_data", type=str, help="Root with REAL and FAKE subfolders")
    ap.add_argument("--files", nargs="*", type=str, help="Optional explicit image files")
    ap.add_argument("--per_group", type=int, default=50, help="Max images per group")
    ap.add_argument("--models", nargs="+", required=True,
                    help="timm model names, e.g., resnet50 xception vit_base_patch16_224")
    ap.add_argument("--target_layer", type=str, default=None,
                    help="Optional dotted layer path for Grad-CAM on CNNs, e.g., layer4.2.conv3")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="cam_out")
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).resolve() if args.data_root else None
    forged_root = Path(args.forged_data).resolve() if args.forged_data else None
    files = [Path(p) for p in args.files] if args.files else None

    groups = collect_image_paths(data_root, forged_root, files, per_group=args.per_group)
    if not groups:
        print("No images found. Provide --data_root or --forged_data or --files.")
        return

    for model_name in args.models:
        print(f"[+] Running CAM for {model_name} on groups: {list(groups.keys())}")
        process_images_for_model(
            model_name=model_name,
            groups=groups,
            out_root=out_root,
            device=device,
            target_layer_str=args.target_layer,
            pretrained=not args.no_pretrained
        )
    print(f"Done. Outputs at {out_root}")

if __name__ == "__main__":
    main()
