# FinalProject: Deepfake and GenAI Model Embedding Analysis

## Overview
This project provides tools for extracting, analyzing, and visualizing image embeddings from various deepfake and generative AI model outputs. The main focus is on:
- Extracting embeddings from image datasets organized by model.
- Visualizing and comparing embeddings using t-SNE and PCA.
- Supporting research and analysis of model fingerprints and dataset structure.

## Directory Structure
```
FinalProject/
├── jobs/                # SLURM job outputs
├── sbatch_files/        # SLURM batch scripts for cluster jobs
├── src/                 # Main source code
│   ├── analyze_embedding_with_pca.py
│   ├── analyze_embedding_with_tsne.py
│   ├── combined_embedding_features_comprations.py
│   ├── compare_embeddings_vs_features.py
│   ├── compare_xception_featuremaps.py
│   ├── extract_embedding.py
│   ├── extract_embedding_multiple.py
│   ├── feature_map_analysis.py
│   ├── gradcam_dynamic.py
│   └── utils/
│       ├── combine_images.py
│       └── plot_dataset_example.py
└── ...
```

## Main Components

### 1. Embedding Extraction (`extract_embedding.py`)
- **Purpose:** Extracts feature embeddings from images using a ResNet-50 backbone.
- **Input:** Directory with subfolders named like `imagenet_..._MODELNAME` containing images in `train/ai/`.
- **Output:** JSON files with image embeddings, one per model.
- **Usage:**
  ```bash
  python src/extract_embedding.py --data_root /path/to/data --out_dir /path/to/output_embeddings
  ```
  - `--data_root`: Root folder containing `imagenet*` project folders.
  - `--out_dir`: Where to save the output JSON files.

### 2. Embedding Visualization (`analyze_embedding_with_tsne.py`)
- **Purpose:** Loads embedding JSONs and visualizes them using t-SNE (or PCA in a similar script).
- **Modes:**
  - `separate`: One t-SNE plot per model.
  - `combined`: All models in one t-SNE plot.
  - `pairs`: Pairwise t-SNE plots for model comparisons.
- **Usage:**
  ```bash
  python src/analyze_embedding_with_tsne.py --embeddings_dir /path/to/output_embeddings --mode combined
  ```
  - See `--help` for all options (sampling, perplexity, output dir, etc).

### 3. Other Analysis Scripts
- `analyze_embedding_with_pca.py`: PCA-based visualization.
- `compare_embeddings_vs_features.py`, `compare_xception_featuremaps.py`: Advanced feature and embedding comparisons.
- `feature_map_analysis.py`, `gradcam_dynamic.py`: Feature map and explainability analysis.

### 4. Utilities
- `utils/combine_images.py`, `utils/plot_dataset_example.py`: Helper scripts for dataset visualization and image combination.

## Example Workflow
1. **Extract Embeddings:**
   ```bash
   python src/extract_embedding.py --data_root /path/to/data --out_dir embeddings
   ```
2. **Visualize with t-SNE:**
   ```bash
   python src/analyze_embedding_with_tsne.py --embeddings_dir embeddings --mode combined
   ```
3. **Explore Other Analyses:**
   - Try PCA, feature map analysis, or pairwise comparisons as needed.

## Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- For GPU acceleration, ensure PyTorch is installed with CUDA support.

## Notes
- Data should be organized as:
  ```
  data_root/
    imagenet_..._MODELNAME/
      train/
        ai/
          *.jpg, *.png, ...
  ```
- Output embeddings are saved as JSON, with each file corresponding to a model.
- t-SNE and PCA plots are saved in the specified output directory.

## SLURM/Cluster Usage
- Use scripts in `sbatch_files/` to run extraction or analysis jobs on a cluster.
- Edit the SBATCH scripts to set paths and resources as needed.

## Contact
For questions or contributions, please contact the project maintainer or open an issue in the repository.
