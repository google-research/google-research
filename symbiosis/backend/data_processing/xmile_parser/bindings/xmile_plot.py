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

"""Plot class."""

import dataclasses
from typing import Optional, Union
import xmile_data_smile
import xmile_entity
import xmile_format
import xmile_globals
import xmile_reset_to
import xmile_scale

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmilePlot:
  """Plot class."""

  class Meta:
    name = "plot"
    namespace = xmile_globals.XMILE_NAMESPACE

    immediately_update_on_user_input: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    color: str = dataclasses.field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    pen_style: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    min: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    max: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    include_in_legend: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    keep_zero_visible: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    pen_width: Optional[Union[int, float]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    index: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    right_axis: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    show_y_axis: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    format: Optional[xmile_format.XmileFormat] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
    entity: Optional[xmile_entity.XmileEntity] = dataclasses.field(
        default=None,
        metadata={
            "name": "entity",
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    scale: Optional[xmile_scale.XmileScale] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
    data: Optional[xmile_data_smile.XmileDataSmile] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    reset_to: Optional[xmile_reset_to.XmileResetTo] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
