# Copyright 2020 The Google Research Authors.
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

#!/bin/bash

set -e
set -x

virtualenv -p python3 env
source env/bin/activate

pip install -r tcc/requirements.txt

# Downloads Pouring data /tmp/pouring_tfrecords/.
tcc/dataset_preparation/download_pouring_data.sh
# Downloads ImageNet pretrained checkpoint (ResNet50v2) to /tmp/
wget -P /tmp/ https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5

# Make empty directory for logs.
mkdir /tmp/alignment_logs
# Copy over demo config to folder.
cp tcc/configs/demo.yml /tmp/alignment_logs/config.yml
# Runs training for 10 iterations on the Pouring dataset.
python -m tcc.train --alsologtostderr --debug
# Evaluates all tasks by embedding 4 videos.
python -m tcc.evaluate --alsologtostderr --max_embs 2 --nocontinuous_eval --visualize
# Extracts embeddings of 4 videos in the val set.
python -m tcc.extract_embeddings --alsologtostderr --logdir /tmp/alignment_logs --dataset pouring --split val --max_embs 2 --keep_data
# Aligns videos using extracted embeddings.
python -m tcc.visualize_alignment --alsologtostderr --video_path /tmp/aligned.mp4  --embs_path /tmp/embeddings.npy
