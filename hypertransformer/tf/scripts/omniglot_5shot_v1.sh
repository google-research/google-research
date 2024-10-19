# Copyright 2024 The Google Research Authors.
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

# Call from the root as
# `./hypertransformer_tf/scripts/omniglot_5shot_v1.sh` with flags
# "--data_cache_dir=<omniglot_cache> --train_log_dir=<output_path>"

./scripts/omniglot_1shot_v1.sh --samples_transformer=100 --samples_cnn=100 $@
