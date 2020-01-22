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

virtualenv -p python3 .
source ./bin/activate

pip install -r uq_benchmark_2019/requirements.txt


# to test the script using a toy dataset

# generate a toy test dataset
python -m uq_benchmark_2019.news.data_generate \
--tr_data_file=./uq_benchmark_2019/news/test_data/train.txt \
--test_data_file=./uq_benchmark_2019/news/test_data/test.txt \
--out_dir=./uq_benchmark_2019/news/test_data \
--vocab_size=1000

# train a toy model using the toy dataset
python -m uq_benchmark_2019.news.classifier \
--num_epochs=1 \
--data_pkl_file=./uq_benchmark_2019/news/test_data/20news_encode_maxlen250_vs1000_in0-2-4-6-8-10-12-14-16-18_trfrac0.9.pkl \
--vocab_size=1000 \
--tr_out_dir=./uq_benchmark_2019/news/test_out


# evaludate the predictions using the toy dataset
python -m uq_benchmark_2019.news.classifier_eval \
--model_dir=./uq_benchmark_2019/news/test_out \
--data_pkl_file=./uq_benchmark_2019/news/test_data/20news_encode_maxlen250_vs1000_in0-2-4-6-8-10-12-14-16-18_trfrac0.9.pkl \
--vocab_size=1000 \
--method=vanilla
