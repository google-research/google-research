#!/bin/bash
# Download LAION400M embeddings and metadata, then build a FAISS index.
#
# Usage:
#   bash setup.sh            # Download all 100 shards (~13 GB total)
#   bash setup.sh 2          # Download only first 2 shards (~260 MB, recommended for testing)
#   bash setup.sh 10         # Download first 10 shards (~1.3 GB, good balance)
#
# Recommendation: start with 2 shards to verify the pipeline works, then
# re-run with a larger number (e.g. 20–100) once you are ready for a full
# experiment.  More shards give a richer retrieval pool and better results.
#
# After downloading, build the FAISS index:
#   pip install autofaiss pyarrow
#   cd laion400m && autofaiss build_index \
#     --embeddings="npy" --index_path="image.index" \
#     --index_infos_path="image_infos.json" --metric_type="ip"

NUM_SHARDS=${1:-2}   # default to 2 shards for a quick test run
LAST_SHARD=$((NUM_SHARDS - 1))

mkdir -p laion400m/metadata laion400m/npy

BASE="https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"

echo "Downloading $NUM_SHARDS metadata shards..."
cd laion400m/metadata/
for i in $(seq 0 $LAST_SHARD); do
  curl -O "${BASE}/metadata/metadata_${i}.parquet"
done
cd -

echo "Downloading $NUM_SHARDS embedding shards..."
cd laion400m/npy
for i in $(seq 0 $LAST_SHARD); do
  curl -O "${BASE}/img_emb/img_emb_${i}.npy"
done
cd -

echo "Done. To build the FAISS index, run:"
echo "  pip install autofaiss pyarrow"
echo "  cd laion400m && autofaiss build_index --embeddings=npy --index_path=image.index --index_infos_path=image_infos.json --metric_type=ip"
