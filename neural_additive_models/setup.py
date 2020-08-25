# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for NAMs."""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'tensorflow>=1.15',
    'numpy>=1.15.2',
    'sklearn',
    'pandas>=0.24',
    'absl-py',
]

nam_description = ('Neural Additive Models: Intepretable ML with Neural Nets')

setup(
    name='neural_additive_models',
    version=0.1,
    description=nam_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/agarwl/google-research/tree/master/neural_additive_models',
    author='Rishabh Agarwal',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='nam, reinforcement, machine, learning, research',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    # package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    # project_urls={},
    license='Apache 2.0',
)
