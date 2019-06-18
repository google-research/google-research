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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r extrapolation/requirements.txt

python -m extrapolation.classifier.classifier_test
python -m extrapolation.influence.eigenvalues_test
python -m extrapolation.utils.dataset_utils_test
python -m extrapolation.utils.running_average_loss_test
python -m extrapolation.utils.tensor_utils_test
python -m extrapolation.vae.run_vae_mnist_test
