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



source 00_vars.sh

# install flask in a venv
apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install flask


cd webarena-homepage
cp templates/index.backup templates/index.html
sed -i "s|SHOPPING_URL|${SHOPPING_URL}|g" templates/index.html
sed -i "s|SHOPPING_ADMIN_URL|${SHOPPING_ADMIN_URL}|g" templates/index.html
sed -i "s|GITLAB_URL|${GITLAB_URL}|g" templates/index.html
sed -i "s|REDDIT_URL|${REDDIT_URL}|g" templates/index.html
sed -i "s|MAP_URL|${MAP_URL}|g" templates/index.html
sed -i "s|WIKIPEDIA_URL|${WIKIPEDIA_URL}|g" templates/index.html

flask run --host=0.0.0.0 --port=$HOMEPAGE_PORT
