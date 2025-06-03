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


#
# Utilities for launch_webviewer.sh
#

################################################################################
## Constants
################################################################################

# Where to store data locally. Paths are with respect to the directory
# containing this script.
DEFAULT_BAKED_SCENE_DIR="baked/example"

# Web viewer config.
DEFAULT_MOUSE_MODE="fps"  # or "orbit"
DEFAULT_QUALITY="high"  # or "phone", "low", or "medium"
DEFAULT_VERTICAL_FOV="40.038544"
DEFAULT_RESOLUTION="1245,825"
DEFAULT_NEAR=0.2
DEFAULT_PORT=8000
DEFAULT_COMBINE_MODE="concat_and_sum"
DEFAULT_USE_DISTANCE_GRID="true"
DEFAULT_USE_BITS="true"

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
    echo "http-server has been installed."
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
  # Override these environment variables if desired.
  local BAKED_SCENE_DIR="${BAKED_SCENE_DIR:-${DEFAULT_BAKED_SCENE_DIR}}"
  local QUALITY="${QUALITY:-${DEFAULT_QUALITY}}"
  local VERTICAL_FOV="${VERTICAL_FOV:-${DEFAULT_VERTICAL_FOV}}"
  local RESOLUTION="${RESOLUTION:-${DEFAULT_RESOLUTION}}"
  local NEAR="${NEAR:-${DEFAULT_NEAR}}"
  local PORT="${PORT:-${DEFAULT_PORT}}"
  local MOUSE_MODE="${MOUSE_MODE:-${DEFAULT_MOUSE_MODE}}"

  # Don't change these unless your training config requires it.
  local COMBINE_MODE="${COMBINE_MODE:-${DEFAULT_COMBINE_MODE}}"
  local USE_DISTANCE_GRID="${USE_DISTANCE_GRID:-${DEFAULT_USE_DISTANCE_GRID}}"
  local USE_BITS="${USE_BITS:-${DEFAULT_USE_BITS}}"

  echo "Open the following link:"
  echo "Link      = http://localhost:${PORT}/"\
"?dir=${BAKED_SCENE_DIR}/sm_000"\
"&quality=${QUALITY}"\
"&combineMode=${COMBINE_MODE}"\
"&s=${RESOLUTION}"\
"&vfovy=${VERTICAL_FOV}"\
"&useDistanceGrid=${USE_DISTANCE_GRID}"\
"&useBits=${USE_BITS}"\
"&mouseMode=${MOUSE_MODE}"
  echo "PWD       = $(pwd)"

  # Launch server with the following arguments.
  # -c-1    : disable caching
  # --gzip  : use *.gz version of a file if possible.
  http-server $(pwd) --gzip -c-1 --port=${PORT}
}
