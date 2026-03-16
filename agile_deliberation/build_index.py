#!/usr/bin/env python3
"""Build a FAISS index from LAION400M .npy embedding shards.

Usage:
    python build_index.py                        # indexes all shards
    python build_index.py --max_shards 1         # only shard 0 (~1 GB)
    python build_index.py --data_dir laion400m --max_shards 1
"""
import argparse
import glob
import json
import os

import faiss
import numpy as np


def build_index(data_dir: str, max_shards: int = None) -> None:
    npy_dir = os.path.join(data_dir, "npy")
    index_path = os.path.join(data_dir, "image.index")
    infos_path = os.path.join(data_dir, "image_infos.json")

    files = sorted(glob.glob(os.path.join(npy_dir, "img_emb_*.npy")))
    if not files:
        raise FileNotFoundError(f"No img_emb_*.npy files found in {npy_dir}")
    if max_shards is not None:
        files = files[:max_shards]

    print(f"Loading {len(files)} shard(s)...")
    chunks = []
    for f in files:
        arr = np.load(f).astype("float32")
        print(f"  {os.path.basename(f)}: {arr.shape}")
        chunks.append(arr)

    all_embs = np.concatenate(chunks, axis=0)
    print(f"Total embeddings: {all_embs.shape}")

    dim = all_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine after L2-norm)
    faiss.normalize_L2(all_embs)
    index.add(all_embs)

    print(f"Writing index to {index_path} ...")
    faiss.write_index(index, index_path)

    infos = {
        "index_key": "Flat",
        "index_param": "metric_type=ip",
        "size in bytes": os.path.getsize(index_path),
        "nb examples": int(all_embs.shape[0]),
        "dimension": dim,
    }
    with open(infos_path, "w") as fp:
        json.dump(infos, fp, indent=2)
    print(f"Index info written to {infos_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="laion400m",
                        help="Directory containing the npy/ subfolder")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Maximum number of shards to index (default: all)")
    args = parser.parse_args()
    build_index(args.data_dir, args.max_shards)
