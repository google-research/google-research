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

set -x
data=$1
seq_len=512
pred_len=96

if [ $data = "ETTm2" ]
then
    python run.py --model tsmixer_rev_in --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.001 --n_block 2 --dropout 0.9 --ff_dim 64
elif [ $data = "weather" ]
then
    python run.py --model tsmixer_rev_in --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 4 --dropout 0.3 --ff_dim 32
elif [ $data = "electricity" ]
then
    python run.py --model tsmixer_rev_in --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 4 --dropout 0.7 --ff_dim 64
elif [ $data = "traffic" ]
then
    python run.py --model tsmixer_rev_in --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate 0.0001 --n_block 8 --dropout 0.7 --ff_dim 64
else
    echo "Unknown dataset"
fi
