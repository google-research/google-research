# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""WT5 setup file."""

import setuptools

setuptools.setup(
    name='wt5',
    version='0.0.0',
    description='Code for "WT5?! Training Text-to-Text Models to Explain their Predictions"',
    author='Anonymous',
    author_email='sharannarang@google.com',
    url='http://none.com',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        'apache-beam',
        't5',
        'langdetect',
        'PyICU',
        'tldextract',
        'spacy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp machinelearning',
)
