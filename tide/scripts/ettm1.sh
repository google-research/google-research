# Copyright 2024 The Google Research Authors.
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
for horizon in 96 192 336 720
  do
    python3 -m train \
    --transform=false \
    --layer_norm=true \
    --holiday=true \
    --dropout_rate=0.5 \
    --batch_size=512 \
    --hidden_size=1024 \
    --num_layers=1 \
    --hist_len=720 \
    --pred_len=$horizon \
    --dataset=ettm1 \
    --decoder_output_dim=8 \
    --final_decoder_hidden=128 \
    --num_split=1 \
    --learning_rate=0.00008393444360049835 \
    --min_num_epochs=0
  done