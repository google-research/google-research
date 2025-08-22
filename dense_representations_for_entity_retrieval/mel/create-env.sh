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


# Creates a virtual environment for running get-mewsli-*.sh data extraction
# scripts.
#
# DEFAULT USAGE (ONLY VIRTUALENV): bash create-env.sh
# Creates a new *virtualenv* and only installs the required Python packages.

# ALTERNATIVE USAGE (WITH CONDA): bash create-env.sh conda
# This method assumes the system has conda installed, and creates a new conda
# environment containing all dependencies needed to run the get-mewsli-*.sh data
# extraction scripts, covering Python itself, some other tools, and the required
# Python packages.
#
# On Linux or Mac, preference is given to environment specifications that
# explicitly list direct and indirect dependencies, with specific versions that
# have been confirmed to work.
#
# For other platforms (or to use as a manual fallback), the provided .yml file
# only gives direct dependencies, with fewer version constraints, and relies
# on conda for resolution.

set -eu

# Environment name.
name=mewsli_env

if [[ $# -eq 1 && $1 == "conda" ]]; then
  # Use conda.
  #
  # Enable calling 'conda activate' inside this script, as documented at
  # https://github.com/conda/conda/issues/7980.
  CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
  source ${CONDA_PATH}/etc/profile.d/conda.sh

  CONDA_CREATE=""
  ADD_PIP_REQS=
  echo "Creating environment ${name}..."
  if [[ "$(uname)" == "Linux" ]]; then
    CONDA_CREATE="conda create --file conda-spec-linux.txt"
    ADD_PIP_REQS=1  # pip-packages still need to be installed.
  elif [[ "$(uname)" == "Darwin" ]]; then
    ADD_PIP_REQS=1  # pip-packages still need to be installed.
    CONDA_CREATE="conda create --file conda-spec-osx.txt"
  else
    ADD_PIP_REQS=0  # pip-packages already covered in .yml file.
    CONDA_CREATE="conda env create --file conda-env.yml"
  fi

  ${CONDA_CREATE} --name ${name}
  conda init bash

  if [[ ${ADD_PIP_REQS} -eq 1 ]]; then
    # Temporarily allow errors and unbound variables in order to work around a
    # bug affecting 'conda activate' for versions prior to 4.10.2.
    # https://github.com/conda/conda/issues/8186#issuecomment-532874667
    set +eu
    conda activate ${name}
    set -eu
    pip install -r "$(dirname $0)/wikinews_extractor/requirements.txt"
  fi
  echo "To activate the conda environment, use 'conda activate ${name}'"
elif [[ $# -eq 0 ]]; then
  # Install pip packages into a new virtualenv. Do not rely on conda.
  virtualenv -p python3 ./${name}
  source ./${name}/bin/activate
  pip install -r "$(dirname $0)/wikinews_extractor/requirements.txt"
  echo "To activate the virtualenv, use 'source ${PWD}/${name}/bin/activate'"
else
  echo "Usage: bash create-env.sh [conda]"
  exit 1
fi