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



# Vertex AI API configuration
export GOOGLE_CLOUD_PROJECT=""
export GOOGLE_CLOUD_LOCATION=""
export GOOGLE_GENAI_USE_VERTEXAI=""


export WA_SHOPPING="http://127.0.0.1:8010"
export WA_SHOPPING_ADMIN="http://127.0.0.1:8020/admin"
export WA_REDDIT="http://127.0.0.1:8030"
export WA_GITLAB="http://127.0.0.1:8040"
export WA_WIKIPEDIA="http://127.0.0.1:8060/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://127.0.0.1:8086"
export WA_HOMEPAGE="http://127.0.0.1:80"

python pipeline_memory.py --website "shopping" --output_dir "./results" --model "gemini-2.5-flash"