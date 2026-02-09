#!/bin/bash
# Copyright 2026 The Google Research Authors.
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



# stop if any error occur
set -e

source 00_vars.sh

# install flask in a venv
apt install python3-venv -y
python3 -m venv venv_reset
source venv_reset/bin/activate

cd reset_server/
python server.py --port ${RESET_PORT} 2>&1 | tee -a server.log
