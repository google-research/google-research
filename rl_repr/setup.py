# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

from setuptools import find_packages
from setuptools import setup

setup(
    name='rl_repr',
    description=(
        'Representation learning for offline reinforcement learning.'
    ),
    packages=find_packages(),
    package_data={},
    install_requires=[
        'tensorflow>=2.2.0',
        'tensorflow-probability>=0.9.0',
        'tf-agents>=0.5.0',
        'gym>=0.17.0',
        'numpy',
        'dm_env',
    ])
