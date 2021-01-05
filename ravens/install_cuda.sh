# Copyright 2021 The Google Research Authors.
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
set -uexo pipefail

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# This is to ensure that apt-get install does not require any user input.
export DEBIAN_FRONTEND=noninteractive

echo "Creating a temporary staging directory..."
readonly TMP_DIR=`mktemp -d --tmpdir ravens-install-dependencies.XXXXXXXXXX`
echo "Created a temporary directory: ${TMP_DIR}"

sudo apt-get update

# Install CUDA (supports Ubuntu 16.04 and 18.04).
if lsb_release -r | grep -q "18.04"; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb -v -O "${TMP_DIR}"/nvidia.deb
    sudo dpkg -i "${TMP_DIR}"/nvidia.deb
    rm "${TMP_DIR}"/nvidia.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda-10-1 cuda-libraries-10-1
    sudo nvidia-xconfig --allow-empty-initial-configuration
else
    # To update to a new version of CUDA tools, open:
    # https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork
    # Right-click the "Download (*.* KB) button and choose "Copy link address" and paste as the URL for wget
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb -v -O "${TMP_DIR}"/nvidia.deb
    sudo dpkg -i "${TMP_DIR}"/nvidia.deb
    rm "${TMP_DIR}"/nvidia.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda-10-1 cuda-libraries-10-1
    sudo nvidia-xconfig --allow-empty-initial-configuration
fi
