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

docker start gitlab
docker start shopping
docker start shopping_admin
docker start forum
# docker start kiwix33
docker start wikipedia

cd openstreetmap-website/
docker compose start

echo -n -e "Waiting 60 seconds for all services to start..."
sleep 60
echo -n -e " done\n"

