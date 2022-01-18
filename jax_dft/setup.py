# coding=utf-8
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

"""Install JAX-DFT."""
import os
import setuptools


# Read in requirements
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
  requirements = [r.strip() for r in f]

setuptools.setup(
    name='jax_dft',
    version='0.0.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=requirements,
    url='https://github.com/google-research/google-research/'
    'tree/master/jax_dft',
    packages=setuptools.find_packages(),
    python_requires='>=3.6')
