# src/analyze_embeddings_tsne.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import itertools
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from tqdm import tqdm


class EmbeddingTSNEAnalyzer:
    def __init__(self,
                 embeddings_dir: Path | str,
                 out_dir: Path | str = "tsne_plots",
                 random_state: int = 42):
        self.embeddings_dir = Path(embeddings_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.model_to_X: Dict[str, np.ndarray] = {}
        self.model_to_keys: Dict[str, List[str]] = {}

    @staticmethod
    def _infer_model_name(stem: str) -> str:
        # Exports looked like: imagenet_ai_0419_biggan__biggan.json
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

    def _run_tsne(self,
                  X: np.ndarray,
                  perplexity: float = 30.0,
                  n_iter: int = 1000,
                  learning_rate: str | float = "auto",
                  init: str = "pca") -> np.ndarray:
        n = X.shape[0]
        safe_perp = min(perplexity, max(5.0, n - 1.0))
        tsne = TSNE(
            n_components=2,
            perplexity=safe_perp,
            n_iter=n_iter,
            learning_rate=learning_rate,
            init=init,
            random_state=self.random_state,
            verbose=0,
            metric="euclidean"
        )
        return tsne.fit_transform(X)

    def _plot_and_save(self,
                       coords: np.ndarray,
                       title: str,
                       out_path: Path) -> None:
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.8)
        plt.title(title, fontsize=14, pad=12)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()

    def analyze_all_separate(self,
                             sample_per_model: Optional[int] = None,
                             tsne_perplexity: float = 30.0,
                             tsne_n_iter: int = 1000,
                             tsne_learning_rate: str | float = "auto",
                             tsne_init: str = "pca") -> List[Tuple[str, Path]]:
        results: List[Tuple[str, Path]] = []
        if not self.model_to_X:
            print("No embeddings loaded. Call load_embeddings() first.")
            return results

        for model, X in tqdm(self.model_to_X.items(), desc="Models"):
            idx = np.arange(X.shape[0])
            if sample_per_model is not None and X.shape[0] > sample_per_model:
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(idx, size=sample_per_model, replace=False)
                X_use = X[idx]
            else:
                X_use = X

            if X_use.shape[0] < 3:
                print(f"Skip {model} - not enough samples ({X_use.shape[0]})")
                continue

            coords = self._run_tsne(
                X_use,
                perplexity=tsne_perplexity,
                n_iter=tsne_n_iter,
                learning_rate=tsne_learning_rate,
                init=tsne_init
            )
            out_png = self.out_dir / f"{model}_tsne.png"
            self._plot_and_save(coords, title=model, out_path=out_png)
            results.append((model, out_png))
        return results

    @staticmethod
    def _assign_colors(n_labels: int) -> List[str]:
        # Use tab20 which has 20 distinct hues, cycle if needed
        cmap = get_cmap("tab20", 20)
        colors = [to_hex(cmap(i % 20)) for i in range(n_labels)]
        return colors

    def analyze_combined(self,
                         sample_per_model: Optional[int] = None,
                         tsne_perplexity: float = 30.0,
                         tsne_n_iter: int = 1000,
                         tsne_learning_rate: str | float = "auto",
                         tsne_init: str = "pca",
                         out_name: str = "all_models_tsne.png") -> Path | None:
        if not self.model_to_X:
            print("No embeddings loaded. Call load_embeddings() first.")
            return None

        # Build concatenated matrix and labels
        models = sorted(self.model_to_X.keys())
        X_blocks = []
        y_labels = []
        per_model_counts = {}

        rng = np.random.default_rng(self.random_state)

        for mi, m in enumerate(models):
            X = self.model_to_X[m]
            idx = np.arange(X.shape[0])
            if sample_per_model is not None and X.shape[0] > sample_per_model:
                idx = rng.choice(idx, size=sample_per_model, replace=False)
            X_sel = X[idx]
            X_blocks.append(X_sel)
            y_labels.append(np.full(X_sel.shape[0], mi, dtype=np.int32))
            per_model_counts[m] = X_sel.shape[0]

        X_all = np.vstack(X_blocks)
        y_all = np.concatenate(y_labels)

        if X_all.shape[0] < 3:
            print("Not enough total samples for t-SNE.")
            return None

        coords = self._run_tsne(
            X_all,
            perplexity=tsne_perplexity,
            n_iter=tsne_n_iter,
            learning_rate=tsne_learning_rate,
            init=tsne_init
        )

        # Assign colors
        colors = self._assign_colors(len(models))
        model_to_color = {m: colors[i] for i, m in enumerate(models)}

        # Plot
        plt.figure(figsize=(8, 6))
        for i, m in enumerate(models):
            mask = (y_all == i)
            plt.scatter(coords[mask, 0], coords[mask, 1],
                        s=6, alpha=0.8, label=m, c=colors[i])

        plt.title("t-SNE of all models", fontsize=14, pad=12)
        plt.xticks([])
        plt.yticks([])

        # Side legend
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

        # Console summary
        print("Combined t-SNE summary:")
        for m in models:
            print(f"- {m}: {per_model_counts[m]} points, color {model_to_color[m]}")
        return out_path

    def analyze_pairs(self,
                      num_pairs: int = 5,
                      pairing_strategy: str = "adjacent",
                      sample_per_model: Optional[int] = None,
                      tsne_perplexity: float = 30.0,
                      tsne_n_iter: int = 1000,
                      tsne_learning_rate: str | float = "auto",
                      tsne_init: str = "pca",
                      subdir_name: str = "pairs") -> List[Tuple[Tuple[str, str], Path]]:
        """
        Create up to num_pairs images. Each image shows a t-SNE of two models plotted together.
        Pairs are chosen either by adjacent names or randomly with fixed seed.
        """
        results: List[Tuple[Tuple[str, str], Path]] = []
        if len(self.model_to_X) < 2:
            print("Need at least two models for pair plots.")
            return results

        models = sorted(self.model_to_X.keys())
        rng = np.random.default_rng(self.random_state)

        pairs: List[Tuple[str, str]]
        if pairing_strategy == "random":
            all_pairs = list(itertools.combinations(models, 2))
            if not all_pairs:
                print("No possible pairs.")
                return results
            rng.shuffle(all_pairs)
            pairs = all_pairs[:num_pairs]
        else:
            # adjacent strategy: pair m0 with m1, m2 with m3, etc.
            take = min(num_pairs * 2, len(models) - (len(models) % 2))
            linear = models[:take]
            pairs = [(linear[i], linear[i + 1]) for i in range(0, len(linear), 2)]
            pairs = pairs[:num_pairs]

        pair_dir = self.out_dir / subdir_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        colors = self._assign_colors(2)
        for a, b in tqdm(pairs, desc="Pairs"):
            Xa = self.model_to_X[a]
            Xb = self.model_to_X[b]

            # sample per model if requested
            def sample_block(X: np.ndarray) -> np.ndarray:
                if sample_per_model is not None and X.shape[0] > sample_per_model:
                    idx = rng.choice(np.arange(X.shape[0]), size=sample_per_model, replace=False)
                    return X[idx]
                return X

            Xa_sel = sample_block(Xa)
            Xb_sel = sample_block(Xb)

            X_all = np.vstack([Xa_sel, Xb_sel])
            if X_all.shape[0] < 3:
                print(f"Skip pair {a} vs {b} - not enough total samples.")
                continue

            coords = self._run_tsne(
                X_all,
                perplexity=tsne_perplexity,
                n_iter=tsne_n_iter,
                learning_rate=tsne_learning_rate,
                init=tsne_init
            )

            na = Xa_sel.shape[0]
            nb = Xb_sel.shape[0]
            mask_a = np.zeros(X_all.shape[0], dtype=bool)
            mask_a[:na] = True
            mask_b = ~mask_a

            plt.figure(figsize=(7, 6))
            plt.scatter(coords[mask_a, 0], coords[mask_a, 1], s=8, alpha=0.85, label=a, c=colors[0])
            plt.scatter(coords[mask_b, 0], coords[mask_b, 1], s=8, alpha=0.85, label=b, c=colors[1])
            plt.title(f"{a} vs {b} - t-SNE", fontsize=14, pad=12)
            plt.xticks([])
            plt.yticks([])
            lgd = plt.legend(title="Model - Color",
                             loc="center left",
                             bbox_to_anchor=(1.02, 0.5),
                             borderaxespad=0.0,
                             frameon=False)
            plt.tight_layout(rect=[0, 0, 0.8, 1])

            safe_a = a.replace("/", "_")
            safe_b = b.replace("/", "_")
            out_path = pair_dir / f"{safe_a}_vs_{safe_b}.png"
            plt.savefig(out_path, dpi=220, bbox_extra_artists=(lgd,), bbox_inches="tight")
            plt.close()

            print(f"Wrote pair plot -> {out_path} ({a}: {na} pts, {b}: {nb} pts)")
            results.append(((a, b), out_path))

        if not results:
            print("No pair plots were created.")
        return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True, type=str,
                    help="Folder with embeddings JSON files")
    ap.add_argument("--out_dir", default="tsne_plots", type=str)
    ap.add_argument("--sample_per_model", default=None, type=int)
    ap.add_argument("--perplexity", default=30.0, type=float)
    ap.add_argument("--n_iter", default=1000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--mode", default="separate",
                    choices=["separate", "combined", "pairs"],
                    help="separate - one t-SNE per model. combined - one t-SNE for all models together. pairs - up to N paired comparisons.")
    # pairs mode extras
    ap.add_argument("--num_pairs", default=5, type=int,
                    help="For pairs mode - number of pair plots to generate")
    ap.add_argument("--pairing_strategy", default="adjacent", choices=["adjacent", "random"],
                    help="How to choose model pairs")
    args = ap.parse_args()

    analyzer = EmbeddingTSNEAnalyzer(
        embeddings_dir=args.embeddings_dir,
        out_dir=args.out_dir,
        random_state=args.seed
    )
    analyzer.load_embeddings()

    if args.mode == "separate":
        made = analyzer.analyze_all_separate(
            sample_per_model=args.sample_per_model,
            tsne_perplexity=args.perplexity,
            tsne_n_iter=args.n_iter
        )
        for model, p in made:
            print(f"Wrote {p} for model {model}")
    elif args.mode == "combined":
        out = analyzer.analyze_combined(
            sample_per_model=args.sample_per_model,
            tsne_perplexity=args.perplexity,
            tsne_n_iter=args.n_iter,
            out_name="all_models_tsne.png"
        )
        if out:
            print(f"Wrote combined plot -> {out}")
    else:
        pairs_made = analyzer.analyze_pairs(
            num_pairs=args.num_pairs,
            pairing_strategy=args.pairing_strategy,
            sample_per_model=args.sample_per_model,
            tsne_perplexity=args.perplexity,
            tsne_n_iter=args.n_iter
        )
        for (a, b), p in pairs_made:
            print(f"Wrote {p} for pair {a} vs {b}")
