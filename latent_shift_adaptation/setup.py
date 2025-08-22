# coding=utf-8
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

"""Package install."""

from setuptools import find_packages
from setuptools import setup


setup(
    name='latent_shift_adaptation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=([
        'numpy>=1.24.1',
        'scikit-learn>=1.2.0',
        'tensorflow>=2.11.0',
        'pandas>=1.5.2',
        'jax>=0.4.1',
        'scipy>=1.10.0',
        'ml_collections',
        'seaborn>0.12.2',
        'git+https://github.com/tsai-kailin/ConditionalOSDE.git'
    ]),
)
