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
chmod +x run.sh

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

cp svhn_data/__init__.py lib/python3.5/site-packages/tensorflow_datasets/image
cp svhn_data/svhn_small.py lib/python3.5/site-packages/tensorflow_datasets/image
cd svhn_data
wget -nc http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -nc http://ufldl.stanford.edu/housenumbers/test_32x32.mat
python gen_svhn_mat.py
cd ..


# Training SVHN from random initialization
bash ./scratch_svhn.sh

# MNIST pre-training
bash ./mnist.sh

# Fine-tuning SVHN from MNIST initialization
bash ./ft.sh

# L2TL on SVHN from MNIST initialization
bash train_l2tl.sh
