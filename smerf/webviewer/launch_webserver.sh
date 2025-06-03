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

 -i
#
# Launches webviewer server on localhost.
#

set -u

# cd into the directory containing this script.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd "${SCRIPT_DIR}"

# Source utility functions and constants
source ./launch_webserver_utils.sh

# Web viewer config. See launch_webviewer_utils.sh for details. For example,
# BAKED_SCENE_DIR="baked/example"
# QUALITY="medium"
# PORT=8000
# RESOLUTION="1280,720"
# MOUSE_MODE="orbit"

# Launch webserver
echo "Installing dependencies..."
install_dependencies

echo "Launching local webserver..."
launch_webviewer
