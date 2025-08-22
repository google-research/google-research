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


import importlib
import sys


if sys.version_info[0] < 3.8:
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True


def is_peft_available():
    return importlib.util.find_spec("peft") is not None


def is_torch_greater_2_0():
    if _is_python_greater_3_8:
        from importlib.metadata import version

        torch_version = version("torch")
    else:
        import pkg_resources

        torch_version = pkg_resources.get_distribution("torch").version
    return torch_version >= "2.0"


def is_diffusers_available():
    return importlib.util.find_spec("diffusers") is not None


def is_bitsandbytes_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_torchvision_available():
    return importlib.util.find_spec("torchvision") is not None


def is_rich_available():
    return importlib.util.find_spec("rich") is not None


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None
