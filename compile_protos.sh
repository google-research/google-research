# Copyright 2022 The Google Research Authors.
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
# Compiles all *.proto files in the repository into *_pb2.py generated files.
#
# `pip install protobuf` should be part of requirements.txt if any of the proto files depend on google.protobuf.

# Exit on fail
set -e

THIRD_PARTY_FOLDER="third_party"
PROTO_COMPILER_CMD="${THIRD_PARTY_FOLDER}/protoc/bin/protoc"
PROTO_COMPILER_VERSION="3.7.1"

# Assert the existence of a folder
assert_folder() {
  # $1: folder to assert
  if [[ ! -d "$1" ]]; then
    mkdir -p "$1"
  fi
}

# Downloads proto compiler (binary)
download_protoc() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  assert_folder protoc
  cd protoc

  # Get and unzip binaries
  wget "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTO_COMPILER_VERSION}/protoc-${PROTO_COMPILER_VERSION}-linux-x86_64.zip"
  unzip "protoc-${PROTO_COMPILER_VERSION}-linux-x86_64.zip"

  # Get license
  wget https://raw.githubusercontent.com/protocolbuffers/protobuf/master/LICENSE

  # Clean
  rm "protoc-${PROTO_COMPILER_VERSION}-linux-x86_64.zip"
  cd ../..
}

# Compile Python protos
compile_protos() {
  for file in $(find ! -path "./${THIRD_PARTY_FOLDER}/*" -name "*.proto"); do
    dir=$(dirname "$file")
    if [[ ! -f "${dir}/__init__.py" ]]; then
      touch "${dir}/__init__.py"
    fi
    ${PROTO_COMPILER_CMD} -I=$dir --python_out=$dir $file
  done
}

download_protoc
compile_protos
