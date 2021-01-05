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

gcloud=../../gcloud

wsize=64
bsize=1
num_gen=1000
num_change=100

# type of energy function
f_type=mlp

# type of base sampler
base_type=rnn

# type of editor
io_enc=mlp

# out func
f_out=identity

data=libpng-$wsize

data_dir=$gcloud/data/fuzz-cooked/$data

gibbs_rounds=1

save_dir=$gcloud/results/$data

export CUDA_VISIBLE_DEVICES=1

python do_fuzz.py \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -window_size $wsize \
    -f_out $f_out \
    -gibbs_rounds $gibbs_rounds \
    -score_func $f_type \
    -base_type $base_type \
    -batch_size $bsize \
    -io_enc $io_enc \
    -gpu 0 \
    -num_change $num_change \
    -num_gen $num_gen \
    -epoch_load -1 \
    $@
