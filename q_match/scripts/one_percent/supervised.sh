#!/bin/bash
# Copyright 2025 The Google Research Authors.
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



# Supervised Training Baseline
# MNIST 1%
# FineTuning Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=mnist_1p \
--eval_task=FTC

# Linear Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=mnist_1p \
--eval_task=LC

# CoverType 1%
# FineTuning Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=covtype_new_1p \
--eval_task=FTC

# Linear Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=covtype_new_1p \
--eval_task=LC

# Higgs 1%
# FineTuning Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=higgs100k1p \
--eval_task=FTC

# Linear Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=higgs100k1p \
--eval_task=LC

# Adult 1%
# FineTuning Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=adult_1p \
--eval_task=FTC

# Linear Classification
python -m q_match.scripts.train \
--logtostderr \
--algo=supervised_training --dataset=adult_1p \
--eval_task=LC
