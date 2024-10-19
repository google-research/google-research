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

### layout model (slower rendering)
# some nice seeds
# 2 5 17 20 1 6 7 11 18 21 22 23 27 
# 38 40 41 44 46 48 60 63 78 90 98 
# 14 26 43 56 58 69 81 88 89 93 99
seed=17 
python video_layout.py --seed $seed

### triplane model (faster rendering) 
# some nice seeds
# 0 2 4 8 9 10  25 26 33 34 37
# 42 51 63 74 84 85 95 97 98
seed=10
python video_triplane.py --seed $seed
