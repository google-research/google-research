# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Install Edward2."""

import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'edward2')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='edward2',
    version='0.0.1',
    description='Edward2',
    author='Edward2 Team',
    author_email='trandustin@google.com',
    url='https://github.com/google-research/google-research/tree/master/simple_probabilistic_programming',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'six',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.9.0',
                       'tensorflow-probability>=0.4.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.9.0',
                           'tensorflow-probability-gpu>=0.4.0'],
        'numpy': ['numpy>=1.7',
                  'scipy>=1.0.0'],
        'tests': [
            'absl-py',
            'pytest',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning probabilistic programming',
)
