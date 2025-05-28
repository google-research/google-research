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
from typing import Optional, Union

import xmile_actions
import xmile_globals
import xmile_image


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileButton:

  class Meta:
    name = "button"
    namespace = xmile_globals.XMILE_NAMESPACE

    z_index: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    background: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    color: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    font_style: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    font_color: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    font_family: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    font_size: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[Union[str, float]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_style: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    transparent: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    corner_radius: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    flat: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    icon_side: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    highlight_on_hover: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    highlight_color: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    label: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    uid: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    x: Optional[Union[float, int]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    y: Optional[Union[float, int]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    width: Optional[Union[float, int]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    height: Optional[Union[int, float]] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    locked: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": xmile_globals.ISEE_NAMESPACE,
        },
    )
    actions: Optional[xmile_actions.XmileActions] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
    image: Optional[xmile_image.XmileImage] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
