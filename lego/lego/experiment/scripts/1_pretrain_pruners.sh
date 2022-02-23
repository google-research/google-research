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

checkpoint_path=$checkpoint_path # ".../logs/FB150k-50/1p.2p.3p.2i.3i-1p.2p.3p.2i.3i.ip.pi/rotate/g-32.0-mode-(Mean,True)-adv-0.5-ngpu-0.1-os-(5000,0,w,u,80)-dataset-(single,3000,e,True,before)-opt-(aggr,adam,cpu,False,5)-sharen-sqrt-lr_step/.../"
data_path=$data_path # .../FB150k-50
question_path=$question_path # .../webqsp
dataset_name=webqsp

CUDA_VISIBLE_DEVICES=0 python ../pretrain_pruners.py --cuda --do_train --do_test \
  --data_path $data_path -n 128 -b 512 -d 1600 -g 32 -a 2.0 -adv \
  --relation_tasks 1p.2p.3p.2i.3i.pi.ip --branch_tasks 2i.3i.pi.2pi \
  -lr 0.00005 --max_steps 200000 --geo rotate --rotate_mode "(Mean,True)" --valid_steps 10000 --log_steps 500 \
  --save_checkpoint_steps 20000 --question_path $question_path \
  --online_sample --cpu_num 2 --online_sample_mode='(5000,0,w,u,80)' --test_batch_size 32 \
  --checkpoint_path $checkpoint_path --print_on_screen \
  --train_online_mode "(single,0,n,False,before)" --optim_mode "(fast,adagrad,gpu,True,5)"
