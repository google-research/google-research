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
# TODO: set these environment variables if using Vertex AI
export GOOGLE_CLOUD_PROJECT=""
export GOOGLE_CLOUD_LOCATION=""
export GOOGLE_GENAI_USE_VERTEXAI=""

mini-extra swebench \
    --model gemini-2.5-flash \
    --subset verified \
    --split test \
    --workers 1 \
    --output ./results_memory_flash \