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
pruner_save_path=$pruner_save_path # "$checkpoint_path/pruners/r-1p.2p.3p.2i.3i.pi.ip-b-2i.3i.pi.2pi/.../"
data_path=$data_path # .../FB150k-50
question_path=$question_path # .../webqsp
dataset_name=webqsp
candidate_path=$candidate_path # "$pruner_save_path/candidates/search_results_Feb"


CUDA_VISIBLE_DEVICES=0 python ../create_candidate_embeddings.py \
  --data_path=$data_path -n 128 -b 512 -d 1600 -g 32 -a 0.5 -adv \
  --task 1p.2p.3p.2i.3i.ip.pi --question_path=$question_path \
  -lr 0.00005 --max_steps 500000 --cpu_num 2 --geo rotate --valid_steps 10000 \
  --checkpoint_path=$checkpoint_path \
  --candidate_path=$candidate_path \
  --rotate_mode="(Mean,True)" --split 'train' --start_idx 0 --end_idx 10000000 --target_rank 5 --target_mrr 0.15 --train_online_mode "(single,0,n,False,before)" --optim_mode "(fast,adagrad,gpu,True,5)"
