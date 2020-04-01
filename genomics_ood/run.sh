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

virtualenv -p python3 .
source ./bin/activate

pip install -r genomics_ood/requirements.txt

DATA_DIR=./genomics_ood/test_data
OUT_DIR=./genomics_ood/test_out
python -m genomics_ood.generative \
--hidden_lstm_size=30 \
--val_freq=1 \
--num_steps=1 \
--in_tr_data_dir=$DATA_DIR/before_2011_in_tr \
--in_val_data_dir=$DATA_DIR/between_2011-2016_in_val \
--ood_val_data_dir=$DATA_DIR/between_2011-2016_ood_val \
--out_dir=$OUT_DIR

python -m genomics_ood.classifier \
--num_motifs=30 \
--val_freq=1 \
--num_steps=1 \
--in_tr_data_dir=$DATA_DIR/before_2011_in_tr \
--in_val_data_dir=$DATA_DIR/between_2011-2016_in_val \
--ood_val_data_dir=$DATA_DIR/between_2011-2016_ood_val \
--label_dict_file=$DATA_DIR/label_dict.json \
--out_dir=$OUT_DIR
