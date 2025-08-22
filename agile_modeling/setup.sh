mkdir laion400m/
mkdir laion400m/train laion400m/eval laion400m/al
mkdir laion400m/train/metadata laion400m/train/npy
mkdir laion400m/eval/metadata laion400m/eval/npy
mkdir laion400m/al/metadata laion400m/al/npy

for i in $(seq 0 99); do
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${i}.parquet laion400m/train/metadata/
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${i}.npy laion400m/train/npy
done
for i in $(seq 100 199); do
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${i}.parquet laion400m/eval/metadata/
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${i}.npy laion400m/eval/npy
done
for i in $(seq 200 299); do
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_${i}.parquet laion400m/al/metadata/
  wget https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${i}.npy laion400m/al/npy
done

# Build FAISS index.
cd laion400m/train
autofaiss build_index --embeddings="npy" --index_path="image.index" --index_infos_path="image_infos.json" --metric_type="ip"
cd -
