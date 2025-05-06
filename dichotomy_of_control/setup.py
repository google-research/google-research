# coding=utf-8
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

from setuptools import find_packages
from setuptools import setup

setup(
    name='dichotomy_of_control',
    version='0.0',
    description=(
        'Dichotomy of Control: Separating What You Can Control from What You Cannot.'
    ),
    packages=find_packages(),
    package_data={},
    install_requires=[
        'gym==0.17.0',
        'numpy==1.23.4',
        'tensorflow-addons==0.18.0',
        'tensorflow-gpu==2.10.0',
        'tf-agents==0.14.0',
        'transformers==4.30.0',
    ])
