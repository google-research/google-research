# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Proto utility module containing helper functions."""

import io
from PIL import Image

from clay.proto import observation_pb2
from clay.utils import view_hierarchy


def fill_observation_proto(observation, vh=None, render_scale=1,
                           include_invisible=False):
  """Fills observation proto using raw bytes of view hierarchy.

  This func assumes these are filled in `observation` beforehand:
  0. image_id
  1. observation.screenshot
  2. arg`vh` OR observation.xml OR observation.json

  Args:
    observation: Observation proto.
    vh: optional, instance of view_hierarchy.ViewHierarch
    render_scale: see ViewHierarchy.get_all_ui_objects()
    include_invisible: bool, whether include invisible nodes.
  """
  assert observation.image_id
  assert observation.screenshot
  assert vh or observation.xml or observation.json
  render_scale = 1
  screenshot = Image.open(io.BytesIO(observation.screenshot))
  width, height = screenshot.size
  observation.screen_width = width
  observation.screen_height = height
  observation.render_scale = render_scale

  if not vh:
    vh = view_hierarchy.ViewHierarchy(screen_width=width, screen_height=height)
    vh.load_json(observation.json)

  ui_objects, bitmap, _, screen_state = vh.get_all_ui_objects(
      scale=render_scale, include_invisible=include_invisible)
  observation.render_bitmap = bitmap.tobytes()
  observation.screen_state.CopyFrom(screen_state)

  for obj in ui_objects:
    bbox = observation_pb2.BoundingBox(
        left=obj.bounding_box.x1,
        right=obj.bounding_box.x2,
        top=obj.bounding_box.y1,
        bottom=obj.bounding_box.y2)
    dom_position = observation_pb2.DomPosition(
        depth=obj.dom_location[0],
        pre_order_id=obj.dom_location[1],
        post_order_id=obj.dom_location[2])
    # Map Python enum to proto enum.
    object_type = str(obj.obj_type).split('.')[1]
    if object_type == 'UNKNOWN':
      object_type = 'UNKNOWN_TYPE'
    object_type = observation_pb2.ObjectType.Value(object_type)
    observation.vh_info.max_children = vh.max_children
    observation.vh_info.pick_from_multiple_roots = vh.pick_from_multiple_roots
    observation.objects.add(
        id=obj.obj_id,
        index=obj.obj_index,
        parent_id=obj.parent_id,
        parent_index=obj.parent_index,
        name=obj.obj_name,
        type=object_type,
        android_class=obj.android_class,
        android_package=obj.android_package,
        text=obj.text,
        content_desc=obj.content_desc,
        resource_id=obj.resource_id,
        clickable=obj.clickable,
        visible=obj.visible,
        enabled=obj.enabled,
        focusable=obj.focusable,
        focused=obj.focused,
        scrollable=obj.scrollable,
        long_clickable=obj.long_clickable,
        selected=obj.selected,
        checkable=obj.checkable,
        checked=obj.checked,
        bbox=bbox,
        grid_location=obj.grid_location.value + 1,
        dom_position=dom_position,
        is_leaf=obj.is_leaf,
        is_actionable=obj.is_actionable(),
    )

