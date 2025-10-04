# src/compare_embeddings_vs_features.py
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ----------------------------
# Helpers - consistent colors and plotting
# ----------------------------
def assign_colors(labels: List[str]) -> Dict[str, str]:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap("tab20", 20)
    return {lab: mcolors.to_hex(cmap(i % 20)) for i, lab in enumerate(labels)}

def plot_2d(Z: np.ndarray, y: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    colors = assign_colors(labels)
    plt.figure(figsize=(8, 6))
    for i, g in enumerate(labels):
        mask = (y == i)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.85, label=g, c=colors[g])
    plt.title(title, fontsize=13)
    plt.xticks([]); plt.yticks([])
    lgd = plt.legend(title="Group - Color", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()

def tsne_2d(X: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    perp = min(perplexity, max(5.0, X.shape[0] - 1.0))
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                learning_rate="auto", init="pca", random_state=seed)
    return tsne.fit_transform(X)

def logreg_cv_acc(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, X, y, cv=cv).mean())

# ----------------------------
# Load embeddings JSONs
# ----------------------------
def infer_group_from_stem(stem: str) -> str:
    if "__" in stem:
        return stem.split("__")[-1]
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def load_embeddings_dir(emb_dir: Path,
                        only_groups: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load per-group embeddings from JSON files in emb_dir.
    - Supports standard files like: imagenet_...__sdv5__xception.json -> group 'sdv5'
    - Supports REAL__xception.json -> group 'REAL'
    - Supports merged file: forged_dataset__REAL_FAKE__*.json (keys prefixed 'REAL/...', 'FAKE/...')
    """
    out: Dict[str, np.ndarray] = {}
    files = sorted([p for p in emb_dir.glob("*.json") if p.is_file()])

    for fpath in files:
        stem = fpath.stem

        # Special case: merged REAL+FAKE file
        if "REAL_FAKE" in stem:
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"[warn] skip {fpath.name} - {e}")
                continue
            # Split by prefix REAL/ or FAKE/
            buckets: Dict[str, List[List[float]]] = {"REAL": [], "FAKE": []}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    if k.startswith("REAL/"):
                        buckets["REAL"].append(v)
                    elif k.startswith("FAKE/"):
                        buckets["FAKE"].append(v)
            for grp, vecs in buckets.items():
                if vecs and (not only_groups or grp in only_groups):
                    out[grp] = np.asarray(vecs, dtype=np.float32)
            continue

        # Normal per-group file
        group = infer_group_from_stem(stem)
        if only_groups and group not in only_groups:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[warn] skip {fpath.name} - {e}")
            continue

        if not isinstance(data, dict) or not data:
            continue

        # keep insertion order
        keys = list(data.keys())
        try:
            X = np.asarray([data[k] for k in keys], dtype=np.float32)
        except Exception as e:
            print(f"[warn] bad vectors in {fpath.name} - {e}")
            continue

        if X.ndim == 2 and X.size > 0:
            out[group] = X

    return out


# ----------------------------
# Load feature maps saved as .npz
# Layout supported:
#   1) feature_root points to model root
#      .../out/xception/<GROUP>/<img>/<stageK>.npz
#   2) feature_root points to parent of the model
#      .../out/<MODEL>/<GROUP>/<img>/<stageK>.npz
# We use GAP over [C, H, W] to get a vector per image
# ----------------------------
def list_feature_paths(feature_root: Path, model_name: str, group: str, layer: str) -> List[Path]:
    """
    Returns list of <...>/<group>/<img>/<layer>.npz
    Works whether feature_root is .../out or .../out/<model_name>.
    """
    # Try model-root directly
    base = feature_root / group
    if not base.exists():
        # Try parent -> model -> group
        base = feature_root / model_name / group
    if not base.exists():
        return []

    out = []
    for img_dir in base.iterdir():
        if img_dir.is_dir():
            f = img_dir / f"{layer}.npz"
            if f.exists():
                out.append(f)
    return sorted(out)

def find_groups_under_model(feature_root: Path, model_name: str) -> List[str]:
    """
    Discover group folders when user passes a model root or its parent.
    """
    # Prefer model-root directly
    base_direct = feature_root
    has_direct_groups = base_direct.exists() and any(p.is_dir() for p in base_direct.iterdir())
    if has_direct_groups and (feature_root / model_name).exists() is False:
        # If path already looks like .../out/xception use it
        return sorted([p.name for p in base_direct.iterdir() if p.is_dir()])

    # Otherwise fallback to parent -> model
    base = feature_root / model_name
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])

def load_gap_feature_vectors(feature_root: Path,
                             model_name: str,
                             layer: str,
                             groups: List[str],
                             max_per_group: int,
                             seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    res: Dict[str, np.ndarray] = {}
    for g in groups:
        fps = list_feature_paths(feature_root, model_name, g, layer)
        if not fps:
            continue
        if len(fps) > max_per_group:
            idx = rng.choice(len(fps), size=max_per_group, replace=False)
            fps = [fps[i] for i in idx]
        X = []
        for f in fps:
            arr = np.load(f)["fmap"]  # [C, H, W]
            X.append(arr.mean(axis=(1, 2)))
        if X:
            res[g] = np.asarray(X, dtype=np.float32)
    return res

# ----------------------------
# Align groups and sample counts between embeddings and features
# ----------------------------
def align_groups(A: Dict[str, np.ndarray],
                 B: Dict[str, np.ndarray],
                 per_group_cap: Optional[int],
                 seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    rng = np.random.default_rng(seed)
    common = sorted(set(A.keys()) & set(B.keys()))
    A2, B2 = {}, {}
    for g in common:
        a = A[g]; b = B[g]
        n = min(len(a), len(b))
        if per_group_cap is not None:
            n = min(n, per_group_cap)
        if n < 3:
            continue
        idx_a = rng.choice(len(a), size=n, replace=False)
        idx_b = rng.choice(len(b), size=n, replace=False)
        A2[g] = a[idx_a]
        B2[g] = b[idx_b]
    groups = sorted(A2.keys())
    return A2, B2, groups

def stack_xy(group_to_X: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(group_to_X.keys())
    Xs, ys = [], []
    for i, g in enumerate(labels):
        Xs.append(group_to_X[g])
        ys.append(np.full(len(group_to_X[g]), i, dtype=np.int32))
    return np.vstack(Xs), np.concatenate(ys), labels

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True, type=str,
                    help="Folder with per-group embeddings JSON files")
    ap.add_argument("--feature_root", required=True, type=str,
                    help="Either .../out or .../out/<model>")
    ap.add_argument("--feature_model", default="xception", type=str,
                    help="Model subfolder to use when feature_root is a parent")
    ap.add_argument("--feature_layer", default="stage3", type=str,
                    help="Which saved layer to use - e.g. stage1 or stage2 or stage3")
    ap.add_argument("--only_groups", nargs="*", default=None,
                    help="Optional subset of groups to include - e.g. REAL FAKE sdv5 biggan")
    ap.add_argument("--sample_per_group", type=int, default=300,
                    help="Cap per group for fairness in both spaces")
    ap.add_argument("--tsne_iter", type=int, default=2000)
    ap.add_argument("--tsne_perp", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="emb_vs_feat_compare")
    args = ap.parse_args()

    emb_dir = Path(args.embeddings_dir).resolve()
    feat_root = Path(args.feature_root).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load embeddings by group
    emb_groups = None
    if args.only_groups:
        emb_groups = [g for g in args.only_groups]
    emb_by_group = load_embeddings_dir(emb_dir, only_groups=emb_groups)
    if not emb_by_group:
        print("No embeddings found for the requested groups.")
        return

    # 2) decide which groups to look for in feature maps
    if args.only_groups:
        groups_for_features = args.only_groups
    else:
        # If not specified, try to use all groups from embeddings, but only keep
        # those that actually exist under the feature root
        discovered = find_groups_under_model(feat_root, args.feature_model)
        groups_for_features = [g for g in sorted(emb_by_group.keys()) if g in set(discovered)]

    if not groups_for_features:
        print("No groups discovered under the feature root for the selected model.")
        return

    # 3) load feature GAP vectors by group for chosen model and layer
    feat_by_group = load_gap_feature_vectors(
        feature_root=feat_root,
        model_name=args.feature_model,
        layer=args.feature_layer,
        groups=groups_for_features,
        max_per_group=args.sample_per_group,
        seed=args.seed
    )
    if not feat_by_group:
        print("No feature maps found for the requested model or layer.")
        return

    # 4) align groups and sample counts between embeddings and features
    emb_aligned, feat_aligned, groups = align_groups(
        emb_by_group, feat_by_group, per_group_cap=args.sample_per_group, seed=args.seed
    )
    if not groups:
        print("No common groups with sufficient samples after alignment.")
        return

    # 5) stack to matrices and labels
    X_emb, y_emb, labels = stack_xy(emb_aligned)
    X_feat, y_feat, labels_feat = stack_xy(feat_aligned)
    assert labels == labels_feat, "internal - labels mismatch"

    # 6) t-SNE for both spaces
    Z_emb = tsne_2d(X_emb, perplexity=args.tsne_perp, n_iter=args.tsne_iter, seed=args.seed)
    Z_feat = tsne_2d(X_feat, perplexity=args.tsne_perp, n_iter=args.tsne_iter, seed=args.seed)

    # 7) plots
    plot_2d(Z_emb, y_emb, labels, title="t-SNE - Embedding space", out_path=out_dir / "tsne_embeddings.png")
    plot_2d(Z_feat, y_feat, labels, title=f"t-SNE - Feature space - {args.feature_model} {args.feature_layer}",
            out_path=out_dir / "tsne_featuremaps.png")

    # 8) simple quantitative comparison - linear separability
    acc_emb = logreg_cv_acc(X_emb, y_emb, cv=5)
    acc_feat = logreg_cv_acc(X_feat, y_feat, cv=5)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Groups: {labels}\n")
        f.write(f"Counts per group - embeddings: {[emb_aligned[g].shape[0] for g in labels]}\n")
        f.write(f"Counts per group - features:   {[feat_aligned[g].shape[0] for g in labels]}\n")
        f.write(f"t-SNE iter: {args.tsne_iter}, perplexity: {args.tsne_perp}\n")
        f.write(f"Linear separability (logreg CV) - embeddings: {acc_emb:.3f}\n")
        f.write(f"Linear separability (logreg CV) - features:   {acc_feat:.3f}\n")

    print("Done.")
    print(f"- Embedding t-SNE: {out_dir / 'tsne_embeddings.png'}")
    print(f"- Feature t-SNE:   {out_dir / 'tsne_featuremaps.png'}")
    print(f"- Summary:         {out_dir / 'summary.txt'}")

if __name__ == "__main__":
    main()
