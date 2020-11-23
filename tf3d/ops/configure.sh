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
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
PIP="pip3"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_ppc64le() {
  [[ "$(uname -m)" == "ppc64le" ]]
}


# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if we are building GPU or CPU ops, default CPU
while [[ "$TF_NEED_CUDA" == "" ]]; do
  read -p "Do you want to build ops again TensorFlow CPU pip package?"\
" Y or enter for CPU (tensorflow-cpu), N for GPU (tensorflow). [Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    [Nn]* ) echo "Build with GPU pip package."; TF_NEED_CUDA=1;;
    "" ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

# Check if we are building against manylinux1 or manylinux2010 pip package,
# default manylinux2010
if is_windows; then
  echo "On windows, skipping toolchain flags.."
  PIP_MANYLINUX2010=0
else
  while [[ "$PIP_MANYLINUX2010" == "" ]]; do
    read -p "Does the pip package have tag manylinux2010 (usually the case for nightly release after Aug 1, 2019, or official releases past 1.14.0)?. Y or enter for manylinux2010, N for manylinux1. [Y/n] " INPUT
    case $INPUT in
      [Yy]* ) PIP_MANYLINUX2010=1;;
      [Nn]* ) PIP_MANYLINUX2010=0;;
      "" ) PIP_MANYLINUX2010=1;;
      * ) echo "Invalid selection: " $INPUT;;
    esac
  done
fi

while [[ "$TF_CUDA_VERSION" == "" ]]; do
  read -p "Are you building against TensorFlow 2.1(including RCs) or newer?[Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build against TensorFlow 2.1 or newer."; TF_CUDA_VERSION=10.1;;
    [Nn]* ) echo "Build against TensorFlow <2.1."; TF_CUDA_VERSION=10.0;;
    "" ) echo "Build against TensorFlow 2.1 or newer."; TF_CUDA_VERSION=10.1;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done


# CPU
if [[ "$TF_NEED_CUDA" == "0" ]]; then

  # Check if it's installed
  if [[ $(${PIP} show tensorflow-cpu) == *tensorflow-cpu* ]] || [[ $(${PIP} show tf-nightly-cpu) == *tf-nightly-cpu* ]] ; then
    echo 'Using installed tensorflow'
  else
    # Uninstall GPU version if it is installed.
    if [[ $(${PIP} show tensorflow) == *tensorflow* ]]; then
      echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
      ${PIP} uninstall tensorflow
    elif [[ $(${PIP} show tf-nightly) == *tf-nightly* ]]; then
      echo 'Already have gpu version of tensorflow installed. Uninstalling......\n'
      ${PIP} uninstall tf-nightly
    fi
    # Install CPU version
    echo 'Installing tensorflow-cpu......\n'
    ${PIP} install tensorflow-cpu
  fi

else

  # Check if it's installed
   if [[ $(${PIP} show tensorflow) == *tensorflow* ]] || [[ $(${PIP} show tf-nightly) == *tf-nightly* ]]; then
    echo 'Using installed tensorflow'
  else
    # Uninstall CPU version if it is installed.
    if [[ $(${PIP} show tensorflow-cpu) == *tensorflow-cpu* ]]; then
      echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
      ${PIP} uninstall tensorflow
    elif [[ $(${PIP} show tf-nightly-cpu) == *tf-nightly-cpu* ]]; then
      echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
      ${PIP} uninstall tf-nightly
    fi
    # Install GPU version
    echo 'Installing tensorflow .....\n'
    ${PIP} install tensorflow
  fi
fi


TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
if [[ "$PIP_MANYLINUX2010" == "0" ]]; then
  write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi
# Add Ubuntu toolchain flags
if is_linux; then
  write_to_bazelrc "build:manylinux2010cuda100 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain"
  write_to_bazelrc "build:manylinux2010cuda101 --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
fi
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"


if is_windows; then
  # Use pywrap_tensorflow instead of tensorflow_framework on Windows
  SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
else
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
fi
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if is_macos; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  elif is_windows; then
    # Use pywrap_tensorflow's import library on Windows. It is in the same dir as the dll/pyd.
    SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi

HEADER_DIR=${TF_CFLAGS:2}
if is_windows; then
  SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR//\\//}
  SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME//\\//}
  HEADER_DIR=${HEADER_DIR//\\//}
fi
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

# TODO(yifeif): do not hardcode path
if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_action_env_to_bazelrc "TF_CUDA_VERSION" ${TF_CUDA_VERSION}
  write_action_env_to_bazelrc "TF_CUDNN_VERSION" "7"
  if is_windows; then
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
  else
    write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
    write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "/usr/local/cuda"
  fi
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
fi

if [[ "$PIP_MANYLINUX2010" == "1" ]]; then
  if [[ "$TF_CUDA_VERSION" == "10.0" ]]; then
    write_to_bazelrc "build --config=manylinux2010cuda100"
    write_to_bazelrc "test --config=manylinux2010cuda100"
  else
    write_to_bazelrc "build --config=manylinux2010cuda101"
    write_to_bazelrc "test --config=manylinux2010cuda101"
  fi
fi
