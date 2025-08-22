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



if [[ $PWD == *"/ev3" ]]; then
  echo "You seem to be in the EV3 folder: PWD = $PWD."
  echo "Moving up to the parent directory."
  cd ..
  echo "PWD = $PWD"
fi

if [[ $PWD != *"/google_research" ]]; then
  echo "You need to run this script from the google_research folder."
  exit 1
fi

# Cf. https://github.com/conda/conda/issues/7980#issuecomment-441358406
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

echo "Installing requirements: this might take some time."
conda env create -f ev3/environment.yaml

echo "Activating the ev3 environment."
conda activate ev3

# Run EV3 on a simple training example.
python -m ev3.run