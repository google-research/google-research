# Copyright 2024 The Google Research Authors.
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
#
# Utilities for launch_webviewer.sh
#

################################################################################
## Constants
################################################################################

# Where to store data locally. Paths are with respect to the directory
# containing this script.
ROOT_SCENES_DIR="scenes"
MIPNERF360_SCENES_SUBDIR="nov_15"

# Web viewer config.
SCENE_NAME="bicycle"
QUALITY="high"
COMBINE_MODE="concat_and_sum"
VERTICAL_FOV="40.038544"
RESOLUTION="1245,825"
NEAR=0.2
USE_DISTANCE_GRID="true"
PORT=8000

################################################################################
## Functions
################################################################################

# Installs http-server using platform-specific tools.
function install_dependencies() {
  # Check if this command exists
  command -v http-server

  # If not, install it.
  if [ $? -eq 1 ]; then
    UNAME_OUTPUTS="$(uname -s)"
    case "${UNAME_OUTPUTS}" in
        Linux*)     install_dependencies_linux;;
        Darwin*)    install_dependencies_osx;;
        *)          echo "Unrecognized platform: ${UNAME_OUTPUTS}"
    esac
    echo ${machine}
  else
    echo "http-server is already installed. Skipping installation..."
  fi
}


# Installs http-server on OSX
function install_dependencies_osx() {
  # Check if this command exists
  command -v brew

  # Install http-server
  if [ $? -eq 0 ]; then
    brew install http-server
  else
   echo "Homebrew is required to install http-server."
   exit 1
  fi
}


# Installs http-server on Linux
function install_dependencies_linux() {
  sudo apt install node-http-server node-opener
}


# Launches web viewer
function launch_webviewer() {
  local DATA_DIR="${ROOT_SCENES_DIR}/${MIPNERF360_SCENES_SUBDIR}/${SCENE_NAME}/sm_000"

  echo "Open the following link:"
  echo "Link      = http://localhost:${PORT}/"\
"?dir=${DATA_DIR}"\
"&quality=${QUALITY}"\
"&combineMode=${COMBINE_MODE}"\
"&s=${RESOLUTION}"\
"&vfovy=${VERTICAL_FOV}"\
"&useDistanceGrid=${USE_DISTANCE_GRID}"
  echo "PWD       = $(pwd)"

  # Launch server with the following arguments.
  # -c-1    : disable caching
  # --gzip  : use *.gz version of a file if possible.
  http-server $(pwd) --gzip -c-1 --port=${PORT}
}
