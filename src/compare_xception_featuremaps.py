import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# -------------------------------------------------
# I/O helpers - assumes your saved layout from the analyzer:
# out/xception/<GROUP>/<image_stem>/<stageK>.npz
# GROUP could be: REAL, FAKE, biggan, glide, sdv5, midjourney, adm, vqdm, wukong, ...
# -------------------------------------------------

def find_groups(xception_root: Path) -> List[str]:
    return sorted([p.name for p in xception_root.iterdir() if p.is_dir()])

def list_feature_paths(xception_root: Path, group: str, layer: str) -> List[Path]:
    base = xception_root / group
    if not base.exists():
        return []
    out = []
    for img_dir in base.iterdir():
        if img_dir.is_dir():
            f = img_dir / f"{layer}.npz"
            if f.exists():
                out.append(f)
    return sorted(out)

def load_feature_npz(fpath: Path) -> np.ndarray:
    # expects key 'fmap' saved via np.savez_compressed
    arr = np.load(fpath)["fmap"]  # shape [C, H, W]
    return arr

def global_avg_pool(fmap: np.ndarray) -> np.ndarray:
    # [C, H, W] -> [C]
    return fmap.mean(axis=(1, 2))

# -------------------------------------------------
# Metrics
# -------------------------------------------------

def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    X, Y: [n, d] feature matrices (rows are samples). Linear CKA.
    """
    Xc = center(X)
    Yc = center(Y)
    # Hilbert-Schmidt inner products
    hsic = (Xc @ Yc.T).sum() ** 2
    x_norm = (Xc @ Xc.T).sum()
    y_norm = (Yc @ Yc.T).sum()
    if x_norm <= 0 or y_norm <= 0:
        return 0.0
    return float(hsic / (x_norm * y_norm))

def separability_logreg(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    scores = cross_val_score(clf, X, y, cv=cv)
    return float(scores.mean())

# -------------------------------------------------
# Plotting
# -------------------------------------------------

def heatmap(mat: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(1.1*len(labels), 0.9*len(labels)))
    im = plt.imshow(mat, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

def scatter_2d(Z: np.ndarray, y: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    # stable palette
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap("tab20", 20)
    colors = [mcolors.to_hex(cmap(i % 20)) for i in range(len(labels))]

    plt.figure(figsize=(8, 6))
    for i, lab in enumerate(labels):
        mask = (y == i)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.8, label=lab, c=colors[i])
    plt.xticks([]); plt.yticks([])
    lgd = plt.legend(title="Group - Color", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()

# -------------------------------------------------
# Core
# -------------------------------------------------

def load_gap_by_group(xc_root: Path, groups: List[str], layer: str, max_per_group: int) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    rng = np.random.default_rng(42)
    group_to_feat: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for g in groups:
        fpaths = list_feature_paths(xc_root, g, layer)
        if not fpaths:
            continue
        if len(fpaths) > max_per_group:
            idx = rng.choice(len(fpaths), size=max_per_group, replace=False)
            fpaths = [fpaths[i] for i in idx]
        X = []
        for f in fpaths:
            fmap = load_feature_npz(f)
            X.append(global_avg_pool(fmap))
        X = np.asarray(X, dtype=np.float32)
        group_to_feat[g] = X
        counts[g] = X.shape[0]
    return group_to_feat, counts

def build_xy(group_to_feat: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(group_to_feat.keys())
    Xs, ys = [], []
    for i, g in enumerate(labels):
        X = group_to_feat[g]
        y = np.full(X.shape[0], i, dtype=np.int32)
        Xs.append(X); ys.append(y)
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    return X_all, y_all, labels

def compute_cka_matrix(group_to_feat: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    labels = sorted(group_to_feat.keys())
    n = len(labels)
    M = np.zeros((n, n), dtype=np.float32)
    for i, gi in enumerate(labels):
        for j, gj in enumerate(labels):
            if j < i: 
                M[i, j] = M[j, i]
                continue
            xi = group_to_feat[gi]; xj = group_to_feat[gj]
            k = min(len(xi), len(xj))
            if k < 2:
                val = 0.0
            else:
                # align sample counts by random subset for fairness
                idx_i = np.random.choice(len(xi), size=k, replace=False)
                idx_j = np.random.choice(len(xj), size=k, replace=False)
                val = linear_cka(xi[idx_i], xj[idx_j])
            M[i, j] = val
            M[j, i] = val
    np.fill_diagonal(M, 1.0)
    return M, labels

def reduce_2d(X: np.ndarray, method: str = "pca", perplexity: float = 30.0, n_iter: int = 1000, seed: int = 42) -> np.ndarray:
    if method.lower() == "tsne":
        tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5.0, X.shape[0]-1.0)),
                    n_iter=n_iter, learning_rate="auto", init="pca", random_state=seed)
        return tsne.fit_transform(X)
    # default pca
    return PCA(n_components=2, random_state=seed).fit_transform(X)

# -------------------------------------------------
# Entry
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xception_out_root", required=True, type=str,
                    help="Folder like out/xception that contains <GROUP>/<img>/stage*.npz")
    ap.add_argument("--layer", default="stage3", choices=["stage1", "stage2", "stage3"],
                    help="Which Xception stage to analyze")
    ap.add_argument("--max_per_group", default=500, type=int,
                    help="Cap samples per group for fairness and speed")
    ap.add_argument("--only_groups", nargs="*", default=None,
                    help="Optional subset of groups to include (e.g. REAL FAKE biggan sdv5)")
    ap.add_argument("--embed_method", default="pca", choices=["pca", "tsne"],
                    help="2-D plot reducer")
    ap.add_argument("--tsne_iter", default=1500, type=int)
    ap.add_argument("--tsne_perp", default=30.0, type=float)
    ap.add_argument("--out_dir", default="xception_compare", type=str)
    args = ap.parse_args()

    xc_root = Path(args.xception_out_root).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    groups = find_groups(xc_root)
    if args.only_groups:
        sel = set([g.lower() for g in args.only_groups])
        groups = [g for g in groups if g.lower() in sel]
    if not groups:
        print("No groups found under xception root.")
        return

    # 1) Load GAP features per group for the chosen layer
    group_to_feat, counts = load_gap_by_group(xc_root, groups, args.layer, args.max_per_group)
    if not group_to_feat:
        print("No feature files found for the requested layer/groups.")
        return

    # 2) Pairwise CKA between groups
    M, labels = compute_cka_matrix(group_to_feat)
    heatmap(M, labels, title=f"Xception {args.layer} - CKA similarity", out_path=out_dir / f"cka_{args.layer}.png")

    # 3) Linear separability (one-vs-rest accuracy)
    X_all, y_all, labels = build_xy(group_to_feat)
    acc = separability_logreg(X_all, y_all, cv=5)
    with open(out_dir / f"readme_{args.layer}.txt", "w", encoding="utf-8") as f:
        f.write(f"Groups: {labels}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"LogReg separability (5-fold CV) using GAP of {args.layer}: {acc:.3f}\n")

    # 4) 2-D embedding of all groups together
    Z = reduce_2d(X_all, method=args.embed_method, perplexity=args.tsne_perp, n_iter=args.tsne_iter)
    scatter_2d(Z, y_all, labels, title=f"Xception {args.layer} - {args.embed_method.upper()} of GAP", out_path=out_dir / f"embed_{args.layer}_{args.embed_method}.png")

    print(f"Done.\n- CKA: {out_dir / f'cka_{args.layer}.png'}\n- 2D: {out_dir / f'embed_{args.layer}_{args.embed_method}.png'}\n- Notes: {out_dir / f'readme_{args.layer}.txt'}")

if __name__ == "__main__":
    main()
