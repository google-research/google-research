# Copyright 2019 The Google Research Authors.
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
chmod +x run.sh

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

cp svhn_data/__init__.py lib/python3.5/site-packages/tensorflow_datasets/image
cp svhn_data/svhn_small.py lib/python3.5/site-packages/tensorflow_datasets/image
cd svhn_data
wget -nc http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -nc http://ufldl.stanford.edu/housenumbers/test_32x32.mat
python gen_svhn_mat.py
cd ..


# Training SVHN from random initialization
python finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=6000 \
    --model_dir=./tmp/l2tl/svhn_small_train_random \
    --train_batch_size=128

python evaluate.py \
    --ckpt_path=./tmp/l2tl/svhn_small_train_random/model.ckpt-6000 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst

# MNIST pre-training
python finetuning.py \
    --target_dataset=mnist \
    --train_steps=10000 \
    --model_dir=./tmp/l2tl/mnist_train \
    --train_batch_size=128

python evaluate.py \
    --ckpt_path=./tmp/l2tl/mnist_train/model.ckpt-10000 \
    --target_dataset=mnist \
    --cls_dense_name=final_dense_dst

# Fine-tuning SVHN from MNIST initialization
python finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=6000 \
    --model_dir=./tmp/l2tl/svhn_small_train_finetune \
    --train_batch_size=128 \
    --warm_start_ckpt_path=./tmp/l2tl/mnist_train/model.ckpt-10000

python evaluate.py \
    --ckpt_path=./tmp/l2tl/svhn_small_train_finetune/model.ckpt-6000 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst


# L2TL on SVHN from MNIST initialization
python train_l2tl.py \
    --train_batch_size=128 \
    --train_steps=6000 \
    --model_dir=./tmp/l2tl/l2tl_train \
    --warm_start_ckpt_path=./tmp/l2tl/mnist_train/model.ckpt-10000

python evaluate.py \
    --ckpt_path=./tmp/l2tl/l2tl_train/model.ckpt-6000 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_target_dense
