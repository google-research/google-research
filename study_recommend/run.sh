#!/bin/bash
# Copyright 2025 The Google Research Authors.
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


# Copyright 2023 The Google Research Authors.
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

virtualenv .
source ./bin/activate

pip install -r study_recommend/requirements.txt

python -m study_recommend.study_recommend --train_data_path=study_recommend/data/student_activity.csv --valid_data_path=study_recommend/data/student_activity.csv --student_info_path=study_recommend/data/student_info.csv --output_path=output --tensorboard_path=output/tensorboard --student_chunk_size=65 --seq_len=65 --per_device_batch_size=2048 --n_steps=5 --learning_rate=0.0016 --model=study --n_recommendations=1,2 --alsologtostderr