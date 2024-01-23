# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
