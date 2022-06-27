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

TASK=$1
AGENT=$2

python demos.py --task=${TASK} --mode=train --n=1000
python demos.py --task=${TASK} --mode=test  --n=100

python train.py --task=${TASK} --agent=${AGENT} --n_demos=1
python train.py --task=${TASK} --agent=${AGENT} --n_demos=10
python train.py --task=${TASK} --agent=${AGENT} --n_demos=100
python train.py --task=${TASK} --agent=${AGENT} --n_demos=1000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=40000

python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=1
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=10
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=100
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=1000
