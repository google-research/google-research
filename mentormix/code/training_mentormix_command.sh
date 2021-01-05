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

# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD/code/"

#############
# CIFAR 10
#############
NL=0.2
ALPHA=2
PERCENTILE=0.8
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar10 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar10/${NL} \
--train_log_dir=cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=0 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--nosecond_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


nohup python code/cifar_eval.py \
  --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=4 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

NL=0.4
ALPHA=8
PERCENTILE=0.6
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar10 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar10/${NL} \
--train_log_dir=cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=1 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--nosecond_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=5 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

NL=0.6
ALPHA=8
PERCENTILE=0.6
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar10 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar10/${NL} \
--train_log_dir=cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=2 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--second_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=6 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


NL=0.8
ALPHA=4
PERCENTILE=0.2
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar10 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar10/${NL} \
--train_log_dir=cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=3 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--second_reweight > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar10 \
  --data_dir=data/cifar10/val/ \
  --checkpoint_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar10_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=7 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

#############
# CIFAR 100
#############

NL=0.2
ALPHA=2
PERCENTILE=0.7
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar100 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar100/${NL} \
--train_log_dir=cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=0 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--nosecond_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


nohup python code/cifar_eval.py \
  --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=4 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


NL=0.4
ALPHA=8
PERCENTILE=0.5
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar100 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar100/${NL} \
--train_log_dir=cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=1 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--nosecond_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=5 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


NL=0.6
ALPHA=4
PERCENTILE=0.3
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar100 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar100/${NL} \
--train_log_dir=cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=2 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--second_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=6 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &


NL=0.8
ALPHA=8
PERCENTILE=0.1
nohup python code/cifar_train_mentormix.py \
--batch_size=128 \
--dataset_name=cifar100 \
--trained_mentornet_dir=mentornet_models/mentornet_pd \
--loss_p_percentile=${PERCENTILE} \
--burn_in_epoch=10 \
--data_dir=data/cifar100/${NL} \
--train_log_dir=cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train \
--studentnet=resnet32 \
--max_number_of_steps=200000 \
--device_id=3 \
--num_epochs_per_decay=30 \
--mixup_alpha=${ALPHA} \
--second_reweight > logs/train_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

nohup python code/cifar_eval.py \
  --dataset_name=cifar100 \
  --data_dir=data/cifar100/val/ \
  --checkpoint_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/train" \
  --eval_dir="cifar100_models/resnet32/${NL}/mentormix_p${PERCENTILE}_a${ALPHA}/eval_val" \
  --studentnet=resnet32 \
  --device_id=7 > logs/eval_${NL}_p${PERCENTILE}_a${ALPHA}.txt &

