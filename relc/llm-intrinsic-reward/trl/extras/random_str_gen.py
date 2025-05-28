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

r"""Generate random experiment ids."""
import random
import string

def generate_experiment_id(length=12):
    # Choose from all the uppercase and lowercase letters and digits
    characters = string.ascii_letters + string.digits
    experiment_id = ''.join(random.choice(characters) for i in range(length))
    return experiment_id
