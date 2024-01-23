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

"""Setup for gminiwob."""

from setuptools import setup

setup(
    name="CoDE",
    version="0.0.1",
    python_requires=">=3.10.8",
    packages=["CoDE"],
    install_requires=[
        "absl-py==1.3.0",
        "gin-config==0.5.0",
        "regex==2022.10.31",
        "gym==0.26.2",
        "tensorflow==2.11.1",
        "dm-sonnet==2.0.0"
    ],
)
