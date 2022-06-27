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
set -e # stop script when there is an error

mkdir ./orig
cd orig

# 1. download airdialogue data
wget https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz

# 2. extract airdialogue data and put it into model directory
tar -xzvf airdialogue_data.tar.gz
mkdir -p ./data/airdialogue/json/
mv ./airdialogue_data/airdialogue/* ./data/airdialogue/json/

mkdir -p ./data/resources/
mv ./airdialogue_data/resources/* ./data/resources

rm -rf ./airdialogue_data
rm -rf ./airdialogue_data.tar.gz

cd ..


