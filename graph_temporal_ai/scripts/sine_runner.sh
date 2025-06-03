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


timestamp() {
  date +"%T"
}
datestamp() {
  date +"%D"
}

t=$(timestamp)
t="$(echo ${t} | tr ':' '-')"

d=$(datestamp)
d=$(echo ${d} | tr '/' '-')

start_time="$d-$t"


# h_dim=1024

num_epochs=20
num_layers=2
batch_size=5
lr=1e-3
l2=1e-2

input_len=50
output_len=50
# save_dir="../lightning_logs/${start_time}/"
save_dir="../lightning_logs/03-03-22-20-51-19/" 


num_nodes_list=( 2 5 10 20 )

for num_nodes in ${num_nodes_list[@]};
do
# echo "number of nodes $num_nodes"
data_dir="sine_${num_nodes}n_10kl_additive"
# echo "data directory ${data_dir}"
mkdir -p $save_dir
# python gen_sine_data.py --num-nodes $num_nodes -d $data_dir &
python lstm_sine_trainer.py --num-nodes $num_nodes -d $data_dir -s $save_dir -e $num_epochs &
# python dcgru_sine_trainer.py -n $num_nodes -d $data_dir -s $save_dir -e $num_epochs &
# python main.py --train-sim-num=$train_sim_num --test-sim-num=$test_sim_num --sim-len=$sim_len \
# --h-dim=$h_dim --n-epochs=$n_epochs --n-layers=$n_layers --batch-size=$batch_size --lr=$lr --l2=$l2 \
# --input-len=$input_len --output-len=$output_len --save-dir=$save_dir & 

cp $(pwd)/sine_runner.sh ${save_dir}/sine_runner.sh

done
