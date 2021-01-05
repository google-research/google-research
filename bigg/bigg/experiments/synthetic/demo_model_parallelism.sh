# Copyright 2021 The Google Research Authors.
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

g_type=grid
ordering=BFS

blksize=100

data_dir=../../../data/$g_type-$ordering

save_dir=../../../scratch/$g_type-$ordering-$blksize

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

# number of gpus to use
num_proc=4

export OMP_NUM_THREADS=2

python -m torch.distributed.launch --nproc_per_node=$num_proc dist_main.py \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -num_proc $num_proc \
    -g_type $g_type \
    -blksize $blksize \
    -epoch_save 100 \
    -bits_compress 256 \
    -gpu 0 \
    $@

