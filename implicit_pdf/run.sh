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
set -e
set -x

python3 -m virtualenv .
source ./bin/activate

pip install -r implicit_pdf/requirements.txt
python -m implicit_pdf.train --mock --symsol_shapes tetX \
--number_training_iterations 2 --batch_size 2 --head_network_specs 16 \
--number_train_queries 100 --number_eval_queries 100 --eval_every 100 \
--number_fourier_components 0 --nosave_models
