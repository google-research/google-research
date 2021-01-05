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

data=mix_line
bsize=6
ctx=15
save_dir=$HOME/scratch/results/ebp/$data-$bsize-$ctx

python3 -m ebp.experiments.main \
    -save_dir $save_dir \
    -data_name $data \
    -batch_size $bsize \
    -num_ctx $ctx \
    -gp_lambda 1 \
    -ent_lam 0.01 \
    -num_epochs 5 \
    -seed 10086 \
    -sigma_eps 1e-1 \
    -beta1 0 \
    $@
