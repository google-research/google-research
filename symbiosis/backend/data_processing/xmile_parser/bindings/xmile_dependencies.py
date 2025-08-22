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

import dataclasses
from typing import List

import xmile_globals
import xmile_var

__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileDependencies:
    class Meta:
        name = "dependencies"
        namespace = xmile_globals.ISEE_NAMESPACE

    var: List[xmile_var.XmileVar] = dataclasses.field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "min_occurs": 1,
        },
    )

