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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r robust_optim/requirements.txt
python -m robust_optim.train --config './robust_optim/config.py'\
  --config.dim 100 --config.num_train 10\
  --config.num_test 400 --config.temperature 0.0001\
  --config.adv.lr 0.1 --config.adv.norm_type linf --config.adv.eps_tot 0.5\
  --config.adv.eps_iter 0.5 --config.adv.niters 100 --config.log_keys\
  '("risk/train/loss","risk/train/zero_one","risk/test/zero_one","risk/train/adv/linf","weight/norm/l1")'\
  --config.optim.name gd --config.optim.lr 0.1 --config.optim.niters 1000 --config.model.arch linear\
  --config.model.regularizer none --config.log_interval 1

