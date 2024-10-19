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

"""Pip setup for https://pypi.org/project/rouge-score/."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rouge_score",
    version="0.1.2",
    author="Google LLC",
    author_email="rouge-opensource@google.com",
    description="Pure python implementation of ROUGE-1.5.5.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/google-research/tree/master/rouge",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "absl-py",
        "nltk",
        "numpy",
        "six>=1.14.0",
    ],
    python_requires=">=3.7",
)
