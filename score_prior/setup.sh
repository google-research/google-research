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



# Install score_sde as a package.
git clone https://github.com/yang-song/score_sde
touch score_sde/setup.py
echo "import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
    name='score_sde',
    packages=['score_sde'],
    package_dir={
      'score_sde': '.',
    },
    install_requires=[
        'jaxlib',
    ],
    python_requires='>=3.8',
)
" >> score_sde/setup.py
cd score_sde
pip install .
cd ..

# controllable_generation.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/controllable_generation.py

# likelihood.py
sed -i 's/from models import/from score_sde.models import/g' score_sde/likelihood.py

# losses.py
sed -i 's/from models import/from score_sde.models import/g' score_sde/losses.py
sed -i 's/from sde_lib import/from score_sde.sde_lib import/g' score_sde/losses.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/losses.py

# run_lib.py
sed -i 's/from models import/from score_sde.models import/g' score_sde/run_lib.py
sed -i 's/from losses import/from score_sde.losses import/g' score_sde/run_lib.py
sed -i 's/from sampling import/from score_sde.sampling import/g' score_sde/run_lib.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/run_lib.py
sed -i 's/from datasets import/from score_sde.datasets import/g' score_sde/run_lib.py
sed -i 's/from evaluation import/from score_sde.evaluation import/g' score_sde/run_lib.py
sed -i 's/from likelihood import/from score_sde.likelihood import/g' score_sde/run_lib.py
sed -i 's/from sde_lib import/from score_sde.sde_lib import/g' score_sde/run_lib.py

# sampling.py
sed -i 's/from sde_lib import/from score_sde.sde_lib import/g' score_sde/sampling.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/sampling.py
sed -i 's/from models.utils import/from score_sde.models.utils import/g' score_sde/sampling.py
sed -i 's/from models import/from score_sde.models import/g' score_sde/sampling.py

# sde_lib.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/sde_lib.py

# models/utils.py
sed -i 's/import sde_lib/from score_sde import sde_lib/g' score_sde/models/utils.py
sed -i 's/from models import/from score_sde.models import/g' score_sde/models/utils.py
sed -i 's/from utils import/from score_sde.utils import/g' score_sde/models/utils.py

# Avoid `AttributeError` with Flax >= 0.6.0.
sed -i 's/optimizer: flax.optim.Optimizer/opt_state: Any/g' score_sde/models/utils.py
sed -i 's/lr: float/params: Any/g' score_sde/models/utils.py
