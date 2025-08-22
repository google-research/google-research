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


CWD=$(pwd)
SOURCE_LANGUAGE=lt
TARGET_LANGUAGE=en


python -m data_tfds_to_tsv \
--tfds_name=wmt19_translate/${SOURCE_LANGUAGE}-${TARGET_LANGUAGE} \
--output_dir=${CWD}/training_data \
--source_language=${SOURCE_LANGUAGE} \
--target_language=${TARGET_LANGUAGE}


python -m flume_make_spm_tfexamples \
--input_path=${CWD}/training_data/ \
--spm_path=${CWD}/training_data/${SOURCE_LANGUAGE}${TARGET_LANGUAGE}.32k.spm.model


python -m train \
--eval_dataset_path=${CWD}/training_data/${SOURCE_LANGUAGE}${TARGET_LANGUAGE}_dev.tfr* \
--guidance_dataset_path=${CWD}/training_data/${SOURCE_LANGUAGE}${TARGET_LANGUAGE}_guide_1percent.tfr* \
--guided_hparam_types=learning_rate \
--model_dir=${CWD}/models/run_$(date +'%m%d_%H') \
--save_checkpoints=True \
--num_layers=2 \
--qkv_dim=128 \
--mlp_dim=128 \
--emb_dim=128 \
--num_heads=2 \
--train_with_guided_parameters=1 \
--training_dataset=wmt_${SOURCE_LANGUAGE}_${TARGET_LANGUAGE} \
--training_dataset_path=${CWD}/training_data/${SOURCE_LANGUAGE}${TARGET_LANGUAGE}_train_99percent.tfr* \
--vocab_path=${CWD}/training_data/${SOURCE_LANGUAGE}${TARGET_LANGUAGE}.32k.spm.model
