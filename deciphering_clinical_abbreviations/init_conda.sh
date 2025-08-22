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



SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -d "${SCRIPT_DIR}/.conda_env" ]; then
  echo "Setting up conda environment."
  conda clean -y --packages --tarballs
  conda env create -f "${SCRIPT_DIR}/linux-env.yml" -p "${SCRIPT_DIR}/.conda_env"
  else echo "Conda environment already exists at ${SCRIPT_DIR}/.conda_env; Skipping setup."
fi

CONDA_INIT_STATUS=$(conda init --all | tail -1)
if [ "$CONDA_INIT_STATUS" != "No action taken." ]; then
  echo "Conda init just reconfigured one of your shells. Please close and restart this shell and rerun the script."
  exit 1
fi
