"""Script to sample high-attention tiles per class and cluster them using the Leiden algorithm.

This pipeline combines CLAM heatmap generation utilities with Leiden-based clustering to
identify representative tile clusters for each class. The workflow follows:

1. Read a dataframe describing slides, cases, and labels.
2. Randomly sample a fixed number of patients per class.
3. For each sampled patient, extract patch features with UNI, compute CLAM attention scores,
   and retain the top-k tiles (top 20% by default).
4. Cluster the retained tile features per class using the Leiden algorithm.
5. Save tile montages per cluster alongside metadata.

The code borrows the feature extraction and attention scoring conventions used in CLAM's
heatmap generation pipeline, while the Leiden clustering stage follows the approach described
in DeepTFtyper (Zhou et al.).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "leidenalg and igraph are required. Please install them in your environment."
    ) from exc

from torch.utils.data import DataLoader

from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from utils.eval_utils import initiate_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TileRecord:
    """Container to track tile-level metadata for clustering."""

    case_id: str
    slide_id: str
    label: str
    coord: Tuple[int, int]
    attention: float
    feature: np.ndarray
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample high-attention tiles per class and cluster them via Leiden"
    )
    parser.add_argument("--df_path", type=str, required=True, help="CSV describing cases")
    parser.add_argument(
        "--case_col", type=str, default="case_id", help="Column containing patient/case ids"
    )
    parser.add_argument(
        "--slide_col", type=str, default="slide_id", help="Column containing slide ids"
    )
    parser.add_argument(
        "--label_col", type=str, default="label", help="Column containing class labels"
    )
    parser.add_argument(
        "--num_patients", type=int, default=10, help="Number of patients per class to sample"
    )
    parser.add_argument(
        "--top_percent", type=float, default=0.2, help="Fraction of top tiles to retain"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--data_h5_dir",
        type=str,
        required=True,
        help="Directory containing patch coordinate h5 files (patches/<slide_id>.h5)",
    )
    parser.add_argument(
        "--slide_dir",
        type=str,
        required=True,
        help="Directory containing raw whole-slide images",
    )
    parser.add_argument("--slide_ext", type=str, default=".svs", help="Slide file extension")
    parser.add_argument(
        "--clam_ckpt", type=str, required=True, help="Path to trained CLAM checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clam_sb",
        choices=["clam_sb", "clam_mb"],
        help="CLAM model variant",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="CLAM size argument (only used for CLAM multi-branch)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=1024,
        help="Embedding dimension (1024 for UNI, 512 for CONCH, etc.)",
    )
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of slide-level classes"
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0.0,
        help="Dropout rate used during training (needed to rebuild the model)",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to store outputs"
    )
    parser.add_argument(
        "--tiles_per_cluster",
        type=int,
        default=40,
        help="Number of representative tiles to save per cluster",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for feature extraction with UNI",
    )
    parser.add_argument(
        "--target_patch_size",
        type=int,
        default=224,
        help="Patch resolution passed to the encoder",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="Optional JSON file mapping label strings to integer indices",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="Number of neighbors when building the KNN graph for Leiden",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter controlling cluster granularity",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="If set, save a UMAP embedding colored by cluster",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (only used when --umap is enabled)",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (only used when --umap is enabled)",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_label_dict(
    labels: Sequence[str], label_map_path: Optional[str]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if label_map_path is not None:
        with open(label_map_path, "r", encoding="utf-8") as handle:
            label_dict = json.load(handle)
        reverse_dict = {int(v): k for k, v in label_dict.items()}
        label_dict = {k: int(v) for k, v in label_dict.items()}
        return label_dict, reverse_dict

    unique = sorted(set(labels))
    label_dict = {lab: idx for idx, lab in enumerate(unique)}
    reverse_dict = {idx: lab for lab, idx in label_dict.items()}
    return label_dict, reverse_dict


def load_slide_dataset(
    h5_path: str, slide_path: str, img_transforms, batch_size: int
) -> DataLoader:
    import openslide

    wsi = openslide.open_slide(slide_path)
    dataset = Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, img_transforms=img_transforms)
    loader_kwargs = {"num_workers": 8, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader


def extract_uni_features(
    loader: DataLoader, encoder: torch.nn.Module
) -> Tuple[torch.Tensor, np.ndarray]:
    features: List[torch.Tensor] = []
    coords: List[np.ndarray] = []

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        coords.append(batch["coord"].numpy())
        with torch.inference_mode():
            feats = encoder(imgs)
        features.append(feats.cpu())

    if not features:
        raise RuntimeError("No patches were loaded for feature extraction.")

    feature_tensor = torch.cat(features, dim=0)
    coord_array = np.concatenate(coords, axis=0)
    return feature_tensor, coord_array


def compute_attention(
    model: torch.nn.Module, features: torch.Tensor, label_index: int
) -> np.ndarray:
    with torch.inference_mode():
        attn = model(features.to(device), attention_only=True)

    if isinstance(model, CLAM_MB):
        if attn.ndim != 2:
            raise ValueError("Unexpected attention tensor shape for CLAM_MB")
        attn = attn[label_index]
    elif isinstance(model, CLAM_SB):
        attn = attn.squeeze(0)
    else:
        raise TypeError("Unsupported model type for attention extraction")

    attn = torch.softmax(attn, dim=-1)
    return attn.cpu().numpy()


def select_top_tiles(
    records: List[TileRecord], top_percent: float
) -> List[TileRecord]:
    if not records:
        return []
    sorted_records = sorted(records, key=lambda rec: rec.attention, reverse=True)
    keep = max(1, int(math.ceil(len(sorted_records) * top_percent)))
    return sorted_records[:keep]


def run_leiden(features: np.ndarray, n_neighbors: int, resolution: float) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Features should be a 2D array")

    if len(features) <= 1:
        return np.zeros(len(features), dtype=int)

    n_neighbors = min(n_neighbors, len(features) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    edges = []
    weights = []
    seen = set()
    for i in range(len(features)):
        for j_idx in range(1, indices.shape[1]):
            j = int(indices[i, j_idx])
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            weight = 1.0 / (1e-8 + distances[i, j_idx])
            weights.append(weight)

    graph = ig.Graph(edges=edges, directed=False)
    graph.add_vertices(len(features) - graph.vcount())
    if weights:
        graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights if weights else None,
        resolution_parameter=resolution,
    )
    membership = np.array(partition.membership, dtype=int)
    return membership


def create_montage(images: Sequence[Image.Image], cols: int = 5) -> Image.Image:
    if not images:
        raise ValueError("No images provided for montage")

    size = images[0].size
    standardized: List[Image.Image] = []
    for img in images:
        if img.size != size:
            img = img.resize(size)
        standardized.append(img)

    rows = math.ceil(len(standardized) / cols)
    montage = Image.new("RGB", (size[0] * cols, size[1] * rows), color=(0, 0, 0))
    for idx, img in enumerate(standardized):
        row = idx // cols
        col = idx % cols
        montage.paste(img, (col * size[0], row * size[1]))
    return montage


def maybe_compute_umap(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_neighbors: int,
    min_dist: float,
) -> None:
    try:
        import umap
    except ImportError:  # pragma: no cover - optional dependency
        print("UMAP not installed; skipping embedding visualization.")
        return

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=0,
    )
    embedding = reducer.fit_transform(features)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=4)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_dir(args.save_dir)

    df = pd.read_csv(args.df_path)
    required_cols = {args.case_col, args.slide_col, args.label_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "Input dataframe must contain columns: " + ", ".join(sorted(required_cols))
        )

    label_dict, _ = build_label_dict(df[args.label_col], args.label_map)

    exp_args = argparse.Namespace(
        model_type=args.model_type,
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        drop_out=args.drop_out,
        n_classes=args.num_classes,
    )
    clam_model = initiate_model(exp_args, args.clam_ckpt, device=device.type)

    encoder, img_transforms = get_encoder(
        "uni_v1", target_img_size=args.target_patch_size
    )
    encoder = encoder.to(device)
    encoder.eval()

    class_to_records: Dict[str, List[TileRecord]] = defaultdict(list)

    for label_name, group in df.groupby(args.label_col):
        cases = group[args.case_col].unique().tolist()
        if not cases:
            continue
        sample_size = min(args.num_patients, len(cases))
        sampled_cases = random.sample(cases, sample_size)

        print(f"Processing label={label_name} with {sample_size} cases")
        for case_id in sampled_cases:
            slides = group[group[args.case_col] == case_id][args.slide_col].tolist()
            if not slides:
                continue

            for slide_id in slides:
                h5_path = os.path.join(args.data_h5_dir, "patches", f"{slide_id}.h5")
                slide_path = os.path.join(args.slide_dir, f"{slide_id}{args.slide_ext}")
                if not os.path.exists(h5_path):
                    print(f"Missing h5 for slide {slide_id}, skipping")
                    continue
                if not os.path.exists(slide_path):
                    print(f"Missing slide file for {slide_id}, skipping")
                    continue

                loader = load_slide_dataset(
                    h5_path=h5_path,
                    slide_path=slide_path,
                    img_transforms=img_transforms,
                    batch_size=args.batch_size,
                )
                features, coords = extract_uni_features(loader, encoder)
                attention_scores = compute_attention(
                    clam_model, features, label_dict[label_name]
                )

                tiles: List[TileRecord] = []
                feature_array = features.cpu().numpy()
                wsi = loader.dataset.wsi  # type: ignore[attr-defined]
                patch_level = loader.dataset.patch_level  # type: ignore[attr-defined]
                patch_size = loader.dataset.patch_size  # type: ignore[attr-defined]
                try:
                    for feat, coord, attn in zip(feature_array, coords, attention_scores):
                        x, y = int(coord[0]), int(coord[1])
                        patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size))
                        patch = patch.convert("RGB")
                        if patch.size != (args.target_patch_size, args.target_patch_size):
                            patch = patch.resize((args.target_patch_size, args.target_patch_size))
                        tiles.append(
                            TileRecord(
                                case_id=case_id,
                                slide_id=slide_id,
                                label=label_name,
                                coord=(x, y),
                                attention=float(attn),
                                feature=feat.astype(np.float32),
                                image=patch,
                            )
                        )
                finally:
                    loader.dataset.wsi.close()  # type: ignore[attr-defined]

                top_tiles = select_top_tiles(tiles, args.top_percent)
                class_to_records[label_name].extend(top_tiles)

    summary_rows = []
    for label_name, records in class_to_records.items():
        if not records:
            continue

        label_dir = os.path.join(args.save_dir, label_name)
        ensure_dir(label_dir)

        features = np.stack([rec.feature for rec in records], axis=0)
        memberships = run_leiden(features, args.n_neighbors, args.resolution)

        if args.umap and len(records) > 1:
            umap_path = os.path.join(label_dir, "umap_clusters.png")
            maybe_compute_umap(
                features,
                memberships,
                umap_path,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )

        cluster_to_tiles: Dict[int, List[TileRecord]] = defaultdict(list)
        for rec, cluster_id in zip(records, memberships):
            cluster_to_tiles[int(cluster_id)].append(rec)

        for cluster_id, tiles in cluster_to_tiles.items():
            cluster_dir = os.path.join(label_dir, f"cluster_{cluster_id}")
            ensure_dir(cluster_dir)

            selected_tiles = tiles
            if len(tiles) > args.tiles_per_cluster:
                selected_tiles = random.sample(tiles, args.tiles_per_cluster)

            montage = create_montage([t.image for t in selected_tiles])
            montage_path = os.path.join(cluster_dir, "montage.jpg")
            montage.save(montage_path)

            tile_df = pd.DataFrame(
                [
                    {
                        "case_id": tile.case_id,
                        "slide_id": tile.slide_id,
                        "label": tile.label,
                        "coord_x": tile.coord[0],
                        "coord_y": tile.coord[1],
                        "attention": tile.attention,
                        "cluster": int(cluster_id),
                    }
                    for tile in tiles
                ]
            )
            tile_df.to_csv(
                os.path.join(cluster_dir, "tile_metadata.csv"), index=False
            )

        summary_rows.extend(
            [
                {
                    "label": rec.label,
                    "case_id": rec.case_id,
                    "slide_id": rec.slide_id,
                    "coord_x": rec.coord[0],
                    "coord_y": rec.coord[1],
                    "attention": rec.attention,
                    "cluster": int(cluster_id),
                }
                for rec, cluster_id in zip(records, memberships)
            ]
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.save_dir, "cluster_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
    else:
        print("No tiles were processed; nothing saved.")


if __name__ == "__main__":
    main()
