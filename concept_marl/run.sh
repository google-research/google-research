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


set -e
set -x

sudo apt-get update && sudo apt update

# create and activate conda env (python 3.9)
eval "$(conda shell.bash hook)"
conda create -y -n concept_marl python=3.9
conda activate concept_marl

# some deps (acme, meltingpot) need
# to be installed manually
cd concept_marl

# acme install
function install_acme {
  git clone https://github.com/deepmind/acme.git
  cd acme
  pip install .[jax,tf,testing,envs]
  cd ..
}

# dmlab2d install (for meltingpot)
function install_dmlab2d() {
  pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl
}

# meltingpot install
function install_meltingpot {
  git clone -b main https://github.com/deepmind/meltingpot
  cd meltingpot
  curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz \
    | tar -xz --directory=meltingpot
  pip install .
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  cd ..
}

# concept marl install
function install_concept_marl {
  cd ..
  pip install -r concept_marl/requirements.txt
}

# install deps
install_acme
install_dmlab2d
install_meltingpot
install_concept_marl

# hit it!
# cooking challenge
python3 -m concept_marl.experiments.run_meltingpot \
--env_name='cooking_basic_mini' \
--episode_length=100 \
--num_steps=5 \
--eval_every=5 \
--batch_size=16

# clean up
python3 -m concept_marl.experiments.run_meltingpot \
--env_name='clean_up_mod_mini' \
--episode_length=100 \
--num_steps=5 \
--eval_every=5 \
--batch_size=16

# CTF
python3 -m concept_marl.experiments.run_meltingpot \
--env_name='capture_the_flag_mod_mini' \
--episode_length=100 \
--num_steps=5 \
--eval_every=5 \
--batch_size=16
