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

nc=10
bsize=1024
enc=rnn
embed=512
num_pub=4
n_rnn=3
act=relu
agg=max
cell=lstm
tok_type=embed
lang=short
data_dir=$gcloud/data/rfill-$lang-$nc

export CUDA_VISIBLE_DEVICES=0

save_dir=$gcloud/results/rfill-q0-$lang-$nc

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_supervised.py \
    -save_dir $save_dir \
    -act_func $act \
    -rlang $lang \
    -io_agg_type $agg \
    -embed_dim $embed \
    -cell_type $cell \
    -numPublicIO $num_pub \
    -data_dir $data_dir \
    -rnn_state_proj False \
    -io_enc $enc \
    -batch_size $bsize \
    -maxNumConcats $nc \
    -rnn_layers $n_rnn \
    -tok_type $tok_type \
    -prog_gen seq \
    -masked False \
    -iter_per_epoch 5000 \
    -epoch_save 1 \
    -num_proc 4 \
    -learning_rate 1e-3 \
    -gpu 0 \
    $@
