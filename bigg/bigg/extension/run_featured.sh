#!/bin/bash
# Copyright 2025 The Google Research Authors.
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



blksize=-1
bsize=32

data_dir=../../data

save_dir=results/blksize-$blksize-b-$bsize

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python main_featured.py \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -blksize $blksize \
  -epoch_save 100 \
  -bits_compress 0 \
  -has_edge_feats=True \
  -has_node_feats=True \
  -batch_size $bsize \
  -num_epochs 1000 \
  -gpu -1 \
  $@
