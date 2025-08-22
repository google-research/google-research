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

"""Common functions for data processing."""

from anytree import AnyNode
from llm4mobile import episode_pb2


def any_tree_to_html(node, layer):
  """Turns an AnyTree representation of view hierarchy into HTML.

  Args:
    node: an AnyTree node.
    layer: which layer is the node in.

  Returns:
    results: output HTML.
  """
  results = ''

  if 'IMAGEVIEW' in node.type:
    node_type = 'img'
  elif 'BUTTON' in node.type:
    node_type = 'button'
  elif 'EDITTEXT' in node.type:
    node_type = 'input'
  elif 'TEXTVIEW' in node.type:
    node_type = 'p'
  else:
    node_type = 'div'
  if node.is_leaf and node.visible:
    html_close_tag = node_type
    results = '<{}{}{}{}> {} </{}>\n'.format(
        node_type,
        ' id={}'.format(node.leaf_id) if node.leaf_id != -1 else '',
        ' class="{}"'.format(' '.join(node.resource_id))
        if node.resource_id
        else '',
        ' alt="{}"'.format(node.content_desc) if node.content_desc else '',
        '{}, '.format(node.text) if node.text else '',
        html_close_tag,
    )
  else:
    children_results = ''
    for child in node.children:
      children_results += any_tree_to_html(child, layer + 1)
      results += children_results

  return results


def parse_episode_proto(episode_proto_string):
  """Parses an episode proto string.

  Args:
    episode_proto_string: the episode proto string.

  Returns:
    screen_html_list: html representation of the screen sequence in the episode.
    node_list: AnyNode representation of the screen sequence in the episode.
    id_to_leaf_id: mapping element indexes to leaf id. Used for grounding eval.
  """
  screen_html_list = []
  screen_node_list = []
  screen_id_map = []
  screen_action_list = []
  package_list = []
  episode = episode_pb2.Episode()
  episode.ParseFromString(episode_proto_string)

  for t in episode.time_steps:
    id_to_leaf_id = {}
    objects = t.observation.objects
    node_list = []
    leaf_count = 0
    for obj in objects:
      package = obj.android_package
      bbox = obj.bbox
      dom = obj.dom_position

      if '/' in obj.resource_id:
        resource_id = obj.resource_id.split('/')[1].split('_')
      else:
        resource_id = obj.resource_id

      if obj.is_leaf and obj.visible:
        leaf_id = leaf_count
        leaf_count += 1
      else:
        leaf_id = -1

      a = AnyNode(
          id=obj.index,
          type=episode_pb2.ObjectType.Name(obj.type),
          bbox=[bbox.left, bbox.top, bbox.right, bbox.bottom],
          grid_location=episode_pb2.GridLocation.Name(obj.grid_location),
          dom_position=[dom.depth, dom.pre_order_id, dom.post_order_id],
          parent_id=obj.parent_index,
          text=obj.text,
          content_desc=obj.content_desc,
          resource_id=resource_id,
          selected=obj.selected,
          clickable=obj.clickable,
          visible=obj.visible,
          enabled=obj.enabled,
          is_leaf=obj.is_leaf,
          leaf_id=leaf_id,
          scrollable=obj.scrollable,
          checkable=obj.checkable,
          checked=obj.checked,
          focusable=obj.focusable,
          focused=obj.focused,
      )
      id_to_leaf_id[obj.index] = leaf_id
      if a.parent_id != -1:
        a.parent = node_list[a.parent_id]
      node_list.append(a)
    screen_action_list.append(id_to_leaf_id[t.action.object.index])
    screen_html_list.append(any_tree_to_html(node_list[0], 0))
    screen_node_list.append(node_list)
    screen_id_map.append(id_to_leaf_id)
    package_list.append(package)
  return (
      screen_html_list,
      screen_node_list,
      screen_action_list,
      screen_id_map,
      package_list,
  )
