mkdir laion400m/
mkdir laion400m/metadata laion400m/npy

cd laion400m/metadata/
for i in $(seq 0 99); do
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${i}.parquet
done
cd -

cd laion400m/npy
for i in $(seq 0 99); do
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${i}.npy
done
cd -

# Build FAISS index.
cd laion400m
autofaiss build_index --embeddings="npy" --index_path="image.index" --index_infos_path="image_infos.json" --metric_type="ip"
cd -
