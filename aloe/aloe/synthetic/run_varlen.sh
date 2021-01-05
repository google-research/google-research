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

data=2spirals
save_dir=$gcloud/results/$data

gibbs_rounds=1
num_steps=16
proposal=geo
bm=gray
num_is=10
f_lr=1
learn=ebm
q_iter=1
lb_type=is
mu0=0.5
bsize=128
learn_stop=True
seed=1023
base=mlp
wclip=-1


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python main_varlen.py \
    -base_type $base \
    -data $data \
    -weight_clip $wclip \
    -f_lr_scale $f_lr \
    -save_dir $save_dir \
    -binmode $bm \
    -proposal $proposal \
    -learn_mode $learn \
    -gibbs_rounds $gibbs_rounds \
    -num_q_steps $num_steps \
    -num_importance_samples $num_is \
    -seed $seed \
    -q_iter $q_iter \
    -mu0 $mu0 \
    -batch_size $bsize \
    -lb_type $lb_type \
    -learn_stop $learn_stop \
    -num_epochs 1001 \
    -epoch_save 50 \
    -gpu 0 \
    $@
