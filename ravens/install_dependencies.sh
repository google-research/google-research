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

#!/bin/bash
set -uexo pipefail

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# This is to ensure that apt-get install does not require any user input.
export DEBIAN_FRONTEND=noninteractive

echo "Creating a temporary staging directory..."
readonly TMP_DIR=`mktemp -d --tmpdir ravens-install-dependencies.XXXXXXXXXX`
echo "Created a temporary directory: ${TMP_DIR}"

sudo apt-get update

# Install Miniconda.
echo "Installing Miniconda (default is Python 3.7)..."
(cd "${TMP_DIR}" && curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
bash "${TMP_DIR}/Miniconda3-latest-Linux-x86_64.sh" -b -u

# Install CUDA (supports Ubuntu 16.04 and 18.04).
if lsb_release -r | grep -q "18.04"; then
    echo "Installing Nvidia CUDA Drivers..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb -v -O "${TMP_DIR}"/nvidia.deb
    sudo dpkg -i "${TMP_DIR}"/nvidia.deb
    rm "${TMP_DIR}"/nvidia.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda cuda-libraries-10-0
    sudo nvidia-xconfig --allow-empty-initial-configuration
else
    echo "Installing Nvidia CUDA Drivers..."
    # To update to a new version of CUDA tools, open:
    # https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork
    # Right-click the "Download (*.* KB) button and choose "Copy link address" and paste as the URL for wget
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb -v -O "${TMP_DIR}"/nvidia.deb
    sudo dpkg -i "${TMP_DIR}"/nvidia.deb
    rm "${TMP_DIR}"/nvidia.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda cuda-libraries-10-0
    sudo nvidia-xconfig --allow-empty-initial-configuration
fi

# A hack to get Tensorflow to link to CUDA 10.2.
# See https://github.com/tensorflow/tensorflow/issues/38194
sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/lib/x86_64-linux-gnu/libcudart.so.10.1

# Add PATH entries into ~/.profile and ~/.bashrc.
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}":/usr/local/go/bin:/usr/local/cuda/bin:/usr/local/cuda/include\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64\n' >> ~/.profile
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}":/usr/local/go/bin:/usr/local/cuda/bin:/usr/local/cuda/include\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64\n' >> ~/.bashrc
source ~/.profile

# Clear Bash cache for python binaries.
hash -r

# Not all packages are available in Miniconda.
echo "Installing pip..."
conda install -y pip

echo "Installing dependencies for Python OpenCV..."
sudo apt-get -y install libgtk2.0-dev
sudo apt-get -y install pkg-config

echo "Install Python libraries..."
./install_python_deps.sh

echo "Installing gdown to download cuDNN files..."
pip install gdown
gdown -O "${TMP_DIR}"/cudnn-10.0-linux-x64-v7.6.4.38.tgz https://drive.google.com/uc?id=19SItH1oSwJX19-Gk76x4x8oHpt2sMzG7

echo "Installing cuDNN..."
tar -C "${TMP_DIR}"/ -xvf "${TMP_DIR}"/cudnn-10.0-linux-x64-v7.6.4.38.tgz
sudo mv "${TMP_DIR}"/cuda/include/* /usr/local/cuda/include/
sudo mv "${TMP_DIR}"/cuda/lib64/* /usr/local/cuda/lib64/
