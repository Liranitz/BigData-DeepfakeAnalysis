# src/analyze_embeddings_pca.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class EmbeddingPCAAnalyzer:
    """
    Loads embeddings JSON files and runs PCA for visualization.
    Modes:
      - separate: one PCA per model, one PNG per model
      - combined: one PCA on all models together with color legend
    """
    def __init__(self,
                 embeddings_dir: Path | str,
                 out_dir: Path | str = "pca_plots",
                 random_state: int = 42,
                 standardize: bool = False):
        self.embeddings_dir = Path(embeddings_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.standardize = standardize

        self.model_to_X: Dict[str, np.ndarray] = {}
        self.model_to_keys: Dict[str, List[str]] = {}

    @staticmethod
    def _infer_model_name(stem: str) -> str:
        # Supports stems like: imagenet_ai_0419_glide__glide
        if "__" in stem:
            return stem.split("__")[-1]
        if "_" in stem:
            return stem.split("_")[-1]
        return stem

    def load_embeddings(self) -> None:
        json_files = sorted(self.embeddings_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {self.embeddings_dir}")
            return

        for jf in json_files:
            model = self._infer_model_name(jf.stem)
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = list(data.keys())
            if not keys:
                continue
            X = np.asarray([data[k] for k in keys], dtype=np.float32)
            if X.ndim != 2:
                print(f"Skip {jf.name} - unexpected shape {X.shape}")
                continue
            self.model_to_X[model] = X
            self.model_to_keys[model] = keys

        print(f"Loaded {len(self.model_to_X)} models from {self.embeddings_dir}")

    @staticmethod
    def _assign_colors(n_labels: int) -> List[str]:
        # 20 distinct hues, cycle if needed
        cmap = get_cmap("tab20", 20)
        return [to_hex(cmap(i % 20)) for i in range(n_labels)]

    def _maybe_standardize(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        # Standardize features to zero mean and unit variance
        return StandardScaler().fit_transform(X)

    def _run_pca(self, X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (coords_2d, explained_variance_ratio)
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        coords = pca.fit_transform(X)
        return coords, pca.explained_variance_ratio_

    def _plot_and_save(self,
                       coords: np.ndarray,
                       title: str,
                       out_path: Path,
                       evr: Optional[np.ndarray] = None) -> None:
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.8)
        if evr is not None and len(evr) >= 2:
            subtitle = f"PC1 {evr[0]*100:.1f}%  •  PC2 {evr[1]*100:.1f}%"
            plt.title(f"{title}\n{subtitle}", fontsize=13, pad=10)
        else:
            plt.title(title, fontsize=14, pad=12)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()

    def analyze_all_separate(self,
                             sample_per_model: Optional[int] = None) -> List[Tuple[str, Path]]:
        results: List[Tuple[str, Path]] = []
        if not self.model_to_X:
            print("No embeddings loaded. Call load_embeddings() first.")
            return results

        rng = np.random.default_rng(self.random_state)

        for model, X in tqdm(self.model_to_X.items(), desc="Models"):
            if sample_per_model is not None and X.shape[0] > sample_per_model:
                idx = rng.choice(X.shape[0], size=sample_per_model, replace=False)
                X_use = X[idx]
            else:
                X_use = X

            if X_use.shape[0] < 3:
                print(f"Skip {model} - not enough samples ({X_use.shape[0]})")
                continue

            X_use = self._maybe_standardize(X_use)
            coords, evr = self._run_pca(X_use, n_components=2)
            out_png = self.out_dir / f"{model}_pca.png"
            self._plot_and_save(coords, title=model, out_path=out_png, evr=evr)
            results.append((model, out_png))
        return results

    def analyze_combined(self,
                         sample_per_model: Optional[int] = None,
                         out_name: str = "all_models_pca.png") -> Path | None:
        if not self.model_to_X:
            print("No embeddings loaded. Call load_embeddings() first.")
            return None

        models = sorted(self.model_to_X.keys())
        rng = np.random.default_rng(self.random_state)

        # Stack data and build labels
        blocks = []
        y = []
        per_model_counts = {}
        for i, m in enumerate(models):
            X = self.model_to_X[m]
            if sample_per_model is not None and X.shape[0] > sample_per_model:
                idx = rng.choice(X.shape[0], size=sample_per_model, replace=False)
                X_sel = X[idx]
            else:
                X_sel = X
            blocks.append(X_sel)
            y.append(np.full(X_sel.shape[0], i, dtype=np.int32))
            per_model_counts[m] = X_sel.shape[0]

        X_all = np.vstack(blocks)
        y_all = np.concatenate(y)

        if X_all.shape[0] < 3:
            print("Not enough total samples for PCA.")
            return None

        X_all = self._maybe_standardize(X_all)
        coords, evr = self._run_pca(X_all, n_components=2)

        colors = self._assign_colors(len(models))

        # Plot with legend sidebar
        plt.figure(figsize=(8, 6))
        for i, m in enumerate(models):
            mask = (y_all == i)
            plt.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.8, c=colors[i], label=m)

        title = "PCA of all models"
        subtitle = f"PC1 {evr[0]*100:.1f}%  •  PC2 {evr[1]*100:.1f}%"
        plt.title(f"{title}\n{subtitle}", fontsize=13, pad=10)
        plt.xticks([])
        plt.yticks([])

        lgd = plt.legend(title="Model - Color",
                         loc="center left",
                         bbox_to_anchor=(1.02, 0.5),
                         borderaxespad=0.0,
                         frameon=False)
        plt.tight_layout(rect=[0, 0, 0.8, 1])

        out_path = self.out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_extra_artists=(lgd,), bbox_inches="tight")
        plt.close()

        print("Combined PCA summary:")
        for m in models:
            print(f"- {m}: {per_model_counts[m]} points")
        return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True, type=str, help="Folder with embeddings JSON files")
    ap.add_argument("--out_dir", default="pca_plots", type=str)
    ap.add_argument("--mode", default="separate", choices=["separate", "combined"],
                    help="separate - one PCA per model. combined - one PCA for all models together.")
    ap.add_argument("--sample_per_model", default=None, type=int, help="Optional cap per model")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--standardize", action="store_true",
                    help="If set, standardize features before PCA")
    args = ap.parse_args()

    analyzer = EmbeddingPCAAnalyzer(
        embeddings_dir=args.embeddings_dir,
        out_dir=args.out_dir,
        random_state=args.seed,
        standardize=args.standardize
    )
    analyzer.load_embeddings()

    if args.mode == "separate":
        made = analyzer.analyze_all_separate(sample_per_model=args.sample_per_model)
        for model, p in made:
            print(f"Wrote {p} for model {model}")
    else:
        out = analyzer.analyze_combined(sample_per_model=args.sample_per_model,
                                        out_name="all_models_pca.png")
        if out:
            print(f"Wrote combined plot -> {out}")
