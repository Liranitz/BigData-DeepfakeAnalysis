import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ----------------------------
# Plot helpers
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

# ----------------------------
# Classifiers
# ----------------------------
def logreg_cv_acc(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, X, y, cv=cv).mean())

def xgboost_cv_acc(X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 42) -> float:
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost") from e
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
        eval_metric="mlogloss", tree_method="hist",
        random_state=seed, n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=skf)
    return float(scores.mean())

# ----------------------------
# Embeddings loader with keys
# ----------------------------
def infer_group_from_stem(stem: str) -> str:
    if "__" in stem:
        return stem.split("__")[-1]
    if "_" in stem:
        return stem.split("_")[-1]
    return stem

def _stem_from_key(k: str) -> str:
    # handle keys like "REAL/img123.png" or just "img123.png"
    name = k.split("/")[-1].split("\\")[-1]
    if "." in name:
        name = ".".join(name.split(".")[:-1])
    return name

def load_embeddings_dir_maps(emb_dir: Path,
                             only_groups: Optional[List[str]] = None
                             ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns: group -> {stem -> vector}
    Supports merged REAL_FAKE files and per-group JSONs.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    files = sorted([p for p in emb_dir.glob("*.json") if p.is_file()])

    for fpath in files:
        stem = fpath.stem

        if "REAL_FAKE" in stem:
            try:
                with fpath.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"[warn] skip {fpath.name} - {e}")
                continue
            buckets: Dict[str, Dict[str, np.ndarray]] = {"REAL": {}, "FAKE": {}}
            for k, v in data.items():
                if not (isinstance(k, str) and isinstance(v, list)):
                    continue
                s = _stem_from_key(k)
                if k.startswith("REAL/"):
                    buckets["REAL"][s] = np.asarray(v, dtype=np.float32)
                elif k.startswith("FAKE/"):
                    buckets["FAKE"][s] = np.asarray(v, dtype=np.float32)
            for grp, d in buckets.items():
                if d and (not only_groups or grp in only_groups):
                    out.setdefault(grp, {}).update(d)
            continue

        group = infer_group_from_stem(stem)
        if only_groups and group not in only_groups:
            continue

        try:
            with fpath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[warn] skip {fpath.name} - {e}")
            continue

        if not isinstance(data, dict) or not data:
            continue

        for k, vec in data.items():
            s = _stem_from_key(k)
            vec_np = np.asarray(vec, dtype=np.float32)
            if vec_np.ndim == 1 and vec_np.size > 0:
                out.setdefault(group, {})[s] = vec_np

    return out

# ----------------------------
# Feature map GAP loader with keys
# Layout: .../<MODEL>/<GROUP>/<img_stem>/<stageK>.npz  or  .../<GROUP>/<img_stem>/<stageK>.npz
# ----------------------------
def list_feature_paths(feature_root: Path, model_name: str, group: str, layer: str) -> List[Path]:
    base = feature_root / group
    if not base.exists():
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

def load_gap_feature_vectors_maps(feature_root: Path,
                                  model_name: str,
                                  layer: str,
                                  groups: List[str],
                                  max_per_group: int,
                                  seed: int = 42
                                  ) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    res: Dict[str, Dict[str, np.ndarray]] = {}
    for g in groups:
        fps = list_feature_paths(feature_root, model_name, g, layer)
        if not fps:
            continue
        if len(fps) > max_per_group:
            idx = rng.choice(len(fps), size=max_per_group, replace=False)
            fps = [fps[i] for i in idx]
        for f in fps:
            # stem is the parent folder name
            stem = f.parent.name
            arr = np.load(f)["fmap"]  # [C, H, W]
            vec = arr.mean(axis=(1, 2)).astype(np.float32)
            res.setdefault(g, {})[stem] = vec
    return res

# ----------------------------
# Build aligned matrices
# ----------------------------
def build_aligned_mats(
    emb_maps: Dict[str, Dict[str, np.ndarray]],
    feat_maps: Dict[str, Dict[str, np.ndarray]],
    per_group_cap: Optional[int],
    seed: int
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]
]:
    """
    Returns:
      emb_by_g[g] -> [N_g, D_e]
      feat_by_g[g] -> [N_g, D_f]
      concat_by_g[g] -> [N_g, D_e + D_f]
      labels -> sorted group names
    Alignment is by intersecting stems inside each group.
    """
    rng = np.random.default_rng(seed)
    groups = sorted(set(emb_maps.keys()) & set(feat_maps.keys()))
    emb_by_g, feat_by_g, concat_by_g = {}, {}, {}

    for g in groups:
        emb_d = emb_maps[g]
        feat_d = feat_maps[g]
        common_stems = sorted(set(emb_d.keys()) & set(feat_d.keys()))
        if not common_stems:
            continue
        if per_group_cap is not None and len(common_stems) > per_group_cap:
            idx = rng.choice(len(common_stems), size=per_group_cap, replace=False)
            common_stems = [common_stems[i] for i in idx]

        Xe, Xf = [], []
        for s in common_stems:
            Xe.append(emb_d[s])
            Xf.append(feat_d[s])
        Xe = np.vstack(Xe).astype(np.float32)
        Xf = np.vstack(Xf).astype(np.float32)
        Xc = np.concatenate([Xe, Xf], axis=1).astype(np.float32)

        emb_by_g[g] = Xe
        feat_by_g[g] = Xf
        concat_by_g[g] = Xc

    labels = sorted(emb_by_g.keys())
    return emb_by_g, feat_by_g, concat_by_g, labels

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
                    help="Optional subset of groups - e.g. REAL FAKE sdv5 biggan")
    ap.add_argument("--sample_per_group", type=int, default=300,
                    help="Cap per group after alignment by stems")
    ap.add_argument("--tsne_iter", type=int, default=2000)
    ap.add_argument("--tsne_perp", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="emb_feat_concat_compare")
    ap.add_argument("--clf", type=str, default="linear", choices=["linear", "xgboost"],
                    help="Classifier for separability score")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    args = ap.parse_args()

    emb_dir = Path(args.embeddings_dir).resolve()
    feat_root = Path(args.feature_root).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load emb maps and feature maps
    emb_maps = load_embeddings_dir_maps(emb_dir, only_groups=args.only_groups)
    if not emb_maps:
        print("No embeddings found for the requested groups.")
        return

    groups_for_features = sorted(emb_maps.keys()) if not args.only_groups else args.only_groups
    feat_maps = load_gap_feature_vectors_maps(
        feature_root=feat_root,
        model_name=args.feature_model,
        layer=args.feature_layer,
        groups=groups_for_features,
        max_per_group=args.sample_per_group,
        seed=args.seed
    )
    if not feat_maps:
        print("No feature maps found for the requested model or layer.")
        return

    # 2) align by stems and build matrices
    emb_by_g, feat_by_g, concat_by_g, labels = build_aligned_mats(
        emb_maps, feat_maps, per_group_cap=args.sample_per_group, seed=args.seed
    )
    if not labels:
        print("No common stems between embeddings and features in any group.")
        return

    # 3) stack matrices
    X_emb, y, labels1 = stack_xy(emb_by_g)
    X_feat, y2, labels2 = stack_xy(feat_by_g)
    X_concat, y3, labels3 = stack_xy(concat_by_g)
    assert labels1 == labels2 == labels3, "Internal labels mismatch"
    labels = labels1

    # 4) t-SNE for visualization
    Z_emb = tsne_2d(X_emb, perplexity=args.tsne_perp, n_iter=args.tsne_iter, seed=args.seed)
    Z_feat = tsne_2d(X_feat, perplexity=args.tsne_perp, n_iter=args.tsne_iter, seed=args.seed)
    Z_concat = tsne_2d(X_concat, perplexity=args.tsne_perp, n_iter=args.tsne_iter, seed=args.seed)

    plot_2d(Z_emb, y, labels, title="t-SNE - Embedding space", out_path=out_dir / "tsne_embeddings.png")
    plot_2d(Z_feat, y2, labels, title=f"t-SNE - Feature space - {args.feature_model} {args.feature_layer}",
            out_path=out_dir / "tsne_featuremaps.png")
    plot_2d(Z_concat, y3, labels, title="t-SNE - Concatenated [embedding ‖ feature] space",
            out_path=out_dir / "tsne_concat.png")

    # 5) separability scores
    if args.clf == "linear":
        acc_emb = logreg_cv_acc(X_emb, y, cv=args.cv)
        acc_feat = logreg_cv_acc(X_feat, y2, cv=args.cv)
        acc_concat = logreg_cv_acc(X_concat, y3, cv=args.cv)
        clf_name = "Logistic Regression"
    else:
        acc_emb = xgboost_cv_acc(X_emb, y, cv=args.cv, seed=args.seed)
        acc_feat = xgboost_cv_acc(X_feat, y2, cv=args.cv, seed=args.seed)
        acc_concat = xgboost_cv_acc(X_concat, y3, cv=args.cv, seed=args.seed)
        clf_name = "XGBoost"

    # 6) summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Groups: {labels}\n")
        f.write(f"Aligned counts per group (by stems): {[emb_by_g[g].shape[0] for g in labels]}\n")
        f.write(f"t-SNE iter: {args.tsne_iter}, perplexity: {args.tsne_perp}\n")
        f.write(f"Classifier: {clf_name}, CV folds: {args.cv}\n")
        f.write(f"Separability - embeddings: {acc_emb:.3f}\n")
        f.write(f"Separability - features:   {acc_feat:.3f}\n")
        f.write(f"Separability - concat:     {acc_concat:.3f}\n")

    print("Done.")
    print(f"- Embedding t-SNE:   {out_dir / 'tsne_embeddings.png'}")
    print(f"- Feature t-SNE:     {out_dir / 'tsne_featuremaps.png'}")
    print(f"- Concatenated t-SNE:{out_dir / 'tsne_concat.png'}")
    print(f"- Summary:           {out_dir / 'summary.txt'} (Classifier: {clf_name})")

if __name__ == "__main__":
    main()
