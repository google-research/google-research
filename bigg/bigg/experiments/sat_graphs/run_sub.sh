# Copyright 2022 The Google Research Authors.
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

g_type=sub
blksize=-1
bsize=2

data_dir=../../../data/sat-$g_type
save_dir=../../..//results/sat-$g_type/blk-$blksize-b-$bsize

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=2

python main.py \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -g_type $g_type \
  -blksize $blksize \
  -epoch_save 500 \
  -bits_compress 256 \
  -accum_grad 5 \
  -batch_size $bsize \
  -gpu 0 \
  $@
