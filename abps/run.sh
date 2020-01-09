# Copyright 2019 The Google Research Authors.
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

virtualenv -p python3 env
source env/bin/activate

pip install -r abps/requirements.txt

python -m abps.train --root_dir=abps/result --game_name=Pong --num_worker=2 --pbt=False --train_steps_per_iteration=10000 --update_policy_iteration=5 --select_policy_way=random --create_hparam=True --num_iterations=1
