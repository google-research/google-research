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

import setuptools


setuptools.setup(
    name='causal_evaluation',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=([
        'numpy>=2.2.6',
        'pandas>=2.2.3',
        'matplotlib>=3.9.1',
        'seaborn>=0.12.2',
        'scikit-learn>=1.6.1',
        'pyarrow>=13.0.0',
        'scipy>=1.14.1',
        'statsmodels>=0.12.2',
        'absl-py>=2.3.0',
        'jax>=0.6.1',
        'folktables>=0.0.12',
        'jupyter'
    ]),
)
