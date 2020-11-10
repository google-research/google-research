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

gcloud=../../gcloud

wsize=64
bsize=512

# type of energy function
f_type=mlp

# type of base sampler
base_type=rnn

# type of editor
io_enc=mlp

# out func
f_out=identity

# scale
f_scale=1

data=libpng-$wsize

data_dir=$gcloud/data/fuzz-cooked/$data

n_q=16
gibbs_rounds=1
num_is=10
mu0=0.1
w_clip=-1

save_dir=$gcloud/results/$data

export CUDA_VISIBLE_DEVICES=0

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_varlen.py \
    -data_dir $data_dir \
    -save_dir $save_dir \
    -window_size $wsize \
    -f_out $f_out \
    -f_scale $f_scale \
    -gibbs_rounds $gibbs_rounds \
    -num_q_steps $n_q \
    -num_importance_samples $num_is \
    -score_func $f_type \
    -base_type $base_type \
    -batch_size $bsize \
    -weight_clip $w_clip \
    -iter_per_epoch 100 \
    -epoch_save 10 \
    -io_enc $io_enc \
    -num_epochs 2001 \
    -mu0 $mu0 \
    -gpu 0 \
    $@
