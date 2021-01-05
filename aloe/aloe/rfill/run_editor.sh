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
bsize=256
data=rfill10k
enc=rnn
embed=512
num_pub=4
n_rnn=3
act=relu
agg=max
cell=lstm
tok_type=embed
lang=short
io_embed_type=normal
data_dir=$gcloud/data/rfill-$lang-$nc

num_is=1
inf_type=sample

export CUDA_VISIBLE_DEVICES=0

save_dir=$gcloud/results/rfill-edit-$lang-$nc

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_editor.py \
    -inf_type $inf_type \
    -num_importance_samples $num_is \
    -save_dir $save_dir \
    -act_func $act \
    -io_embed_type $io_embed_type \
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
    -iter_per_epoch 500 \
    -num_proc 4 \
    -epoch_save 1 \
    -learning_rate 1e-3 \
    -epoch_load -1 \
    -gpu 0 \
    $@
