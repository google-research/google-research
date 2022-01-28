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

"""Library to clean and annotate view hierarchy."""

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple

import anytree
from apache_beam import metrics
import tensorflow as tf

from clay.utils import anytree_utils
from clay.utils import example_utils
from clay.utils import image_utils
from clay.utils import layout_utils
from clay.utils import tfexample_tree

# Define some classes with special rules:
DRAWER_LABEL_NAME = 'DRAWER'
WEB_VIEW_LABEL = 'WEB_VIEW'
AD_LABEL = 'ADVERTISEMENT'
ON_OFF_LABEL = 'SWITCH'
RADIO_LABEL = 'RADIO_BUTTON'
CHECKBOX_LABEL = 'CHECKBOX'
IMAGE_LABEL = 'IMAGE'
VECTOR_IMAGE_LABEL = 'PICTOGRAM'
CONTAINER_LABEL = 'CONTAINER'
CONTAINER_ADDITIONAL_LABEL = 'CONTAINER_ADDITIONAL'  # aligned containers
TEXT_BUTTON_LABEL = 'BUTTON'
NAVIGATION_BAR_ADDITIONAL_LABEL = 'NAVIGATION_BAR_ADDITIONAL'  # top/bottom bar

# Common string patterns found on keyboards.
KEYBOARD_STRS = [
    'qwertyuiop', 'asdfghjkl', 'zxcvbnm', '?123', '123456789', '2abc', '3def',
    '4ghi', '5jkl', '6mno', '7pqrs', '8tuv', '9wxyz'
]

# Keys - labels, values - parent class names for the label
NAME_CLASSES_MAPPING_PARENT = {
    'DRAWER': ['DrawerLayout'],
    'LIST_ITEM': [
        'ListView', 'RecyclerView', 'ListPopUpWindow', 'TabItem', 'GridView'
    ],
    'RADIO_BUTTON': ['RadioGroup'],
}

NAME_CLASSES_MAPPING_CHILDREN = {
    'NAVIGATION_BAR': ['TabView', 'TabItem', 'TabLayout$']
}

# Keys - labels, values - class names for the label
NAME_CLASSES_MAPPING = {
    'IMAGE': ['ImageView', 'SquaredImageView', 'RoundedImageView', 'VideoView'],
    'PICTOGRAM': [
        'Icon', 'IconButton', 'GlyphView', 'ActionMenuItemView',
        'BottomNavigationItemView', 'ZoomButton', 'FloatingActionButton',
        'OverflowMenuButton', 'SquareImageViewIcon'
    ],
    'BUTTON': [
        'Button', 'CheckedTextView', 'SquareButton', 'MaterialButton',
        'ImageButton', 'CheckableImageButton', 'AppCompatButton',
        'AppCompatImageButton', 'TabView', 'TabItem', 'TabLayout'
    ],
    'TEXT': [
        'FormsTextView', 'CustomTextView', 'MaterialTextView', 'RMTextView',
        'AutoResizeTextView', 'CustomFontTextView', 'DialogTitle',
        'BaseTextView', 'TextViewCustomFont', 'TextView'
    ],
    'TEXT_INPUT': [
        'EditText', 'TextInput', 'TextInputEditText', 'AutoCompleteTextView',
        'TextInputLayout', 'TextInputView', 'FormsEditText', 'SearchBoxView',
        'AppCompatAutoCompleteTextView', 'UrlInputView'
    ],
    'MAP': ['SupportMapFragment', 'MapView'],
    'CHECKBOX': ['CheckBox'],
    'SWITCH': [
        'Switch', 'ToggleButton', 'SwitchCompat', 'CompoundButton',
        'SwitchButton'
    ],
    'PAGER_INDICATOR': [
        'ViewPagerIndicatorDots', 'PageIndicator', 'CircleIndicator',
        'PagerIndicator'
    ],
    'RADIO_BUTTON': ['RadioButton'],
    'SLIDER': ['SeekBar'],
    'SPINNER': ['Spinner'],
    'PROGRESS_BAR': ['ProgressBar', 'SeekBar', 'Seekbar', 'VerticalSeekbar'],
    'ADVERTISEMENT': ['AdView', 'HtmlBannerWebView', 'AdContainer'],
    'NAVIGATION_BAR': [
        'ButtonBar', 'BottomNavigationView', 'TabBarBottomMenuItemView',
        'BottomBar', 'SlidingTab', 'BottomBarTab', 'FixedBottomNavigationTab',
        'MainViewBottomTab', 'TabButton', 'MainTabView', 'ActionBarTabGroup',
        'BottomBarTabView', 'BottomTabGroupView', 'TabItemView'
    ],
    'TOOLBAR': ['ToolBar', 'TitleBar', 'ActionBar'],
    'LIST_ITEM': [
        'LeftMenuItemView', 'MenuItemView', 'BaseListViewItem',
        'NavigationMenuItemView'
    ],
    'CARD_VIEW': [
        'CardView', 'MaterialCardView', 'TileView', 'AlertDialogLayout'
    ],
    'CONTAINER': [
        'ListView', 'RecyclerView', 'ListPopUpWindow', 'TabItem', 'GridView',
        'MenuDropDownListView', 'DropDownListView', 'DropDownList',
        'RadioGroup', 'CoordinatorLayout', 'TableLayout'
    ],
    'DATE_PICKER': ['DatePicker', 'CalendarView'],
    'NUMBER_STEPPER': ['NumberPicker'],
    'CONTAINER_ADDITIONAL': ['Layout', 'View'],  # not in the final class list
    'WEB_VIEW': ['WebView']  # not in the final class list
}

# Resource ID mappings.
RES_ID_BOTTOM_BAR = 'com.android.systemui:id/ends_group'
RES_ID_MAPPING = {
    'android:id/navigationBarBackground':
        'NAVIGATION_BAR',
    'android:id/statusBarBackground':
        'NAVIGATION_BAR',
    'com.android.systemui:id/status_bar_container':
        NAVIGATION_BAR_ADDITIONAL_LABEL,
    RES_ID_BOTTOM_BAR:
        'NAVIGATION_BAR',
    'com.google.android.apps.maps:id/map_frame':
        'MAP',
    'android.support.design.chip.Chip':
        'BUTTON',
    'com.samsung.android.honeyboard:id/layout_honeyboard':
        'KEYBOARD',
    'com.google.android.inputmethod.latin:id/keyboard_holder':
        'KEYBOARD'
}

# Containers.
CONTAINERS = [
    'DRAWER', 'ADVERTISEMENT', 'TOOLBAR', 'NAVIGATION_BAR', 'LIST_ITEM',
    'CARD_VIEW', 'CONTAINER', 'KEYBOARD'
]

# define preference order of labels - manually defined, from specific to general
LABELS_SORTED = [
    'PICTOGRAM', 'LIST_ITEM', 'LABEL', 'DRAWER', 'RADIO_BUTTON',
    'ADVERTISEMENT', 'MAP', 'NUMBER_STEPPER', 'SWITCH', 'SLIDER', 'TOOLBAR',
    'NAVIGATION_BAR', 'CARD_VIEW', 'CHECKBOX', 'DATE_PICKER', 'IMAGE',
    'TEXT_INPUT', 'BUTTON', 'PAGER_INDICATOR', 'TEXT', 'PROGRESS_BAR',
    'SPINNER', 'WEB_VIEW', 'CONTAINER', 'CONTAINER_ADDITIONAL', 'ROOT',
    'BACKGROUND', None
]

# define elements that can be associated with a text label
LABELABLE_ELEMENTS = ['RADIO_BUTTON', 'SWITCH', 'CHECKBOX']

# define labels that can not take large proportion of the screen
LABELS_MEDIUM_TO_SMALL = [
    'PICTOGRAM', 'BUTTON', 'CHECKBOX', 'SWITCH', 'PAGER_INDICATOR',
    'RADIO_BUTTON', 'SLIDER', 'SPINNER', 'PROGRESS_BAR', 'NAVIGATION_BAR',
    'TOOLBAR'
]


@dataclasses.dataclass()
class CleaningOptions:
  """Cleaning options."""
  class_to_label: Optional[Dict[str, int]] = None
  label_to_class: Optional[Dict[int, str]] = None
  drawer_max_area: float = 0.9
  medium_small_max_area: float = 0.5
  # If True, do not delete boxes, but label them as BACKGROUND instead.
  keep_all_boxes: bool = True
  run_cleaning: bool = True
  run_labeling: bool = True


def _remove_boxes_behind_occluder(root,
                                  set_as_bg = False):
  """Remove boxes that are behind an occluder."""
  # Note: we need to perform pre-order traversal, as it is the order in which
  # the elements are rendered.
  nodes = list(anytree.PreOrderIter(root))
  for i, node in enumerate(nodes):
    for occluder in nodes[i + 1:]:
      if occluder.opacity > 0.8 and occluder not in node.descendants and image_utils.overlap(
          node.bbox, occluder.bbox) > 0.:
        layout_utils.cut_overlapping_box(
            node, occluder.bbox, set_as_bg=set_as_bg)


def _sort_labels_by_generality(
    labels_matched):
  """Sort labels according to the manually specified order."""
  order_dict = {label: i for i, label in enumerate(LABELS_SORTED)}
  try:
    sorted_labels = [
        cls_label
        for cls_label in sorted(labels_matched, key=lambda x: order_dict[x[1]])
    ]
  except KeyError:
    sorted_labels = []
    metrics.Metrics.counter('semantics',
                            'label not in ordered labels list').inc()
  return sorted_labels


def _select_label(matched_labels,
                  node):
  """Collision resolution: deciding which labels to use if muliple options."""
  if not matched_labels:
    return '', ''

  # dict of form {'class match': label} with one label per class match.
  matched_labels_dict = {}
  for class_labels_pair in matched_labels:
    if len(class_labels_pair[0]) > 1:
      # in case we will define mapping w/ collisions like in RICO.
      metrics.Metrics.counter('semantics', 'PROBLEM_many_label_per_class').inc()
    if class_labels_pair[1] not in matched_labels_dict:
      # only 2nd child can be a drawer.
      index_child = anytree_utils.get_node_index(node)
      if index_child != 1 and class_labels_pair[0][0] == DRAWER_LABEL_NAME:
        continue
      matched_labels_dict[class_labels_pair[1]] = class_labels_pair[0][0]

  # no match.
  if not matched_labels_dict:
    return '', ''

  # just one candidate match.
  if len(matched_labels_dict) == 1:
    cls, label = list(matched_labels_dict.items())[0]
    return label, cls

  # matched multiple classes (due to substring match or ancestors).
  labels_matched = list(matched_labels_dict.items())
  sorted_labels_matched = _sort_labels_by_generality(labels_matched)

  cls, label = sorted_labels_matched[0]

  return label, cls


def _get_labels_from_parent(
    node):
  """Check labels dependent on parent."""
  if node.parent is None:
    return []

  parent_cls = node.parent.class_name

  # check labels dependent on parents.
  class_name_mapping = layout_utils.flip_dict(NAME_CLASSES_MAPPING_PARENT)
  return layout_utils.find_str_in_dict(parent_cls, class_name_mapping)


def _get_labels_from_children(
    node):
  """Check labels dependent on children."""
  matched_labels = []
  class_name_mapping = layout_utils.flip_dict(NAME_CLASSES_MAPPING_CHILDREN)
  for child in node.children:
    matched_labels.extend(
        layout_utils.find_str_in_dict(child.class_name, class_name_mapping))
  return matched_labels


def _get_label_name(node):
  """Infer label from view hierarchy node.

  Process is as follows (in order).
    1. Infer from node's resource_id.
    2. Infer from node's class_name.
    1. Try to match class name to applicable labels.
    2. Check parent class to identify if it is a list/drawer item.

  Args:
    node: node of interest for which labels is needed

  Returns:
    label assignment
  """
  class_name = node.class_name
  resource_id = node.resource_id
  # Identify label from resource_id (detects status bar/bottom navigation bar,
  # some buttons, maps from resource_id).
  if resource_id in RES_ID_MAPPING:
    return RES_ID_MAPPING[resource_id]

  class_name_mapping = layout_utils.flip_dict(NAME_CLASSES_MAPPING)
  matched_labels = layout_utils.find_str_in_dict(class_name, class_name_mapping)

  # check labels dependent on parents and children.
  matched_labels.extend(_get_labels_from_parent(node))
  matched_labels.extend(_get_labels_from_children(node))

  name, key = _select_label(matched_labels, node)

  if (name == RADIO_LABEL or name == CHECKBOX_LABEL) and node.text:
    # Radio button that has text field is not checkbox.
    name = 'BUTTON'

  if key == 'TextView'.lower():
    if not node.is_clickable:
      name = 'TEXT'
    else:
      name = 'BUTTON'
  if (not name or name == 'CONTAINER_ADDITIONAL') and node.text:
    name = 'TEXT'

  return name


def _remove_overlapping_boxes(root,
                              threshold = 0.95,
                              set_as_bg = False):
  """Delete boxes that overlap by more than threshold."""

  nodes = list(anytree.PreOrderIter(root))
  for index, node in enumerate(nodes):
    # Get all nodes overlaping with the current node, including itself.
    over = [
        n for n in nodes[index:]
        if (image_utils.iou(node.bbox, n.bbox) >= threshold)
    ]
    if over:
      # Sort the overlapping nodes by label specificity, node with most specific
      # label will be the last one. If all the nodes have the same labels, the
      # order doesn't change as python sorting is stable. In either case, we
      # keep the last node as it has the most specific label or covers previous
      # nodes in pre-order traversal order during rendering.
      over.sort(
          key=lambda n: LABELS_SORTED.index(n.bbox.ui_class), reverse=True)
      for o in over[:-1]:
        # Only keep the last one.
        layout_utils.delete_node(o, set_as_bg=set_as_bg)
        metrics.Metrics.counter('semantics', 'PROBLEM: duplicate').inc()


def _clean_webview_adview_bboxes(root,
                                 class_to_label,
                                 img_aspect_ratio,
                                 set_as_bg = False):
  """Clean up webview/advertisement conflicts."""
  # Update ADs made with WEB_VIEW.
  for node in anytree.PreOrderIter(root):
    box = node.bbox
    # Common aspect ratio for narrow ad: 16:9 or 1.91:1 (GoogleAds recommended).
    if box.ui_label == WEB_VIEW_LABEL and box.aspect_ratio * img_aspect_ratio < 2 and box.area < 0.3:
      node.set_labels(AD_LABEL, class_to_label[AD_LABEL])

  webview_nodes = layout_utils.get_nodes_by_class(root, WEB_VIEW_LABEL)
  onoff_nodes = layout_utils.get_nodes_by_class(root, ON_OFF_LABEL)

  # Catch webview and on/off label overlay - should be an ad.
  for webview in webview_nodes:
    for onoff in onoff_nodes:
      if image_utils.iou(webview.bbox, onoff.bbox) > 0.5:
        # Webview that should be an ad.
        webview.set_labels(AD_LABEL, class_to_label[AD_LABEL])
        layout_utils.delete_node(onoff, set_as_bg=set_as_bg)


def _select_containers(root,
                       class_to_label,
                       set_as_bg = False):
  """Select container boxes that have more than 2 children, remove the others."""
  for container in layout_utils.get_nodes_by_class(
      root, [CONTAINER_ADDITIONAL_LABEL, CONTAINER_LABEL])[::-1]:
    if len(container.children) < 2:
      layout_utils.delete_node(container, set_as_bg=set_as_bg)
      continue
    container.set_labels('CONTAINER', class_to_label[CONTAINER_LABEL])


def _relabel_images_in_bottom_bar(root,
                                  class_to_label):
  """Make sure images inside bottom bar are labelled as pictograms.

  Images in the bottom bar (such as the back button and the home button are
  considered to be pictograms.

  Args:
    root: Root node of the screenshot, represented as anytree.AnyNode.
    class_to_label: Mapping that defines the labels of individual classes.
  """
  for node in layout_utils.get_nodes_by_class(root, ['NAVIGATION_BAR']):
    if (node.bbox.bottom >= 0.95 and node.bbox.height < 0.1 and
        node.resource_id == RES_ID_BOTTOM_BAR):
      for child in layout_utils.get_nodes_by_class(root, ['IMAGE']):
        child.set_labels('PICTOGRAM', class_to_label['PICTOGRAM'])


def _clean_text(root,
                class_to_label,
                set_as_bg = False):
  """Remove text boxes that do not have text and relabel special characters."""
  for node in [
      n for n in anytree.PreOrderIter(root) if n.bbox.ui_class == 'TEXT'
  ]:
    text = node.text.strip()
    if not text:
      layout_utils.delete_node(node, set_as_bg=set_as_bg)
    if len(text) > 1:
      continue
    # Checkbox/rectangle unicode characters.
    # (see https://jrgraphix.net/r/Unicode/2600-26FF)
    if text in ['', '\u2610', '\u2611', '\u2612']:
      node.set_labels('CHECKBOX', class_to_label['CHECKBOX'])
    # Radio button unicode characters.
    elif text in ['\u1F518', '\u2299', '\u25C9', '\u25CE']:
      node.set_labels('RADIO_BUTTON', class_to_label['RADIO_BUTTON'])
    # Range of unicode symbols/icons.
    elif '\u20D0' <= text <= '\u29FF' or '\uE000' <= text <= '\uF8FF' or text <= '\u1F300':
      node.set_labels('PICTOGRAM', class_to_label['PICTOGRAM'])


def _convert_to_label(root, class_to_label):
  """Convert text/container boxes to labels if they are next to specific UI elements."""
  for txt_node in [
      n for n in anytree.PreOrderIter(root)
      if n.bbox.ui_class in ['CONTAINER', 'TEXT'] or n.text
  ]:
    for ui_element in layout_utils.get_nodes_by_class(root, LABELABLE_ELEMENTS):
      if txt_node.parent == ui_element.parent:
        txt_node.set_labels('LABEL', class_to_label['LABEL'])


def clean_screen_tfexample(
    example,
    options):
  """Clean view hierarchy bboxes and attempts to infer classes from nodes.

  Args:
    example: input view hierarchy in tf.Example format.
    options: cleaning options.

  Returns:
    Cleaned view hierarchy in tf.Example format or None if the Example is
    considered not clean.
  """
  example = copy.copy(example)
  keep_all_boxes = options.keep_all_boxes

  image = example_utils.get_image_pil(example)
  width, height = image.size
  img_aspect_ratio = width / height
  tree = tfexample_tree.tfexample_to_tree(example)

  if tree is None:
    return None

  # VH information is removed so that it can be replaced by inferred classes.
  tfexample_tree.delete_hierarchy_nodes(example)

  # Local (node-by-node) analysis.
  # Remark, we need to convert the iterator to a list, otherwise the node
  # deletion interferes with the iterator.
  for node in reversed(list(anytree.PreOrderIter(tree))):
    bbox = node.bbox
    if options.run_cleaning:
      if bbox.width <= 0 or bbox.height <= 0:
        # bbox size is invalid.
        metrics.Metrics.counter('semantics', 'zero_width_or_height').inc()
        layout_utils.delete_node(node, set_as_bg=keep_all_boxes)
        continue

      if (bbox.aspect_ratio * img_aspect_ratio > 100 or
          bbox.aspect_ratio * img_aspect_ratio < 0.01 or bbox.area < 0.0001):
        # bbox invalid, too narrow, too small.
        metrics.Metrics.counter('semantics', 'invalid_box_too_small').inc()
        layout_utils.delete_node(node, set_as_bg=keep_all_boxes)
        continue

      # Discard invisible nodes that do not have visible descendants.
      has_visible_descendant = any(
          [n.visible_to_user for n in node.descendants])
      if not node.visible_to_user and not has_visible_descendant:
        layout_utils.delete_branch(node, set_as_bg=keep_all_boxes)
        continue

      # Discard Samsung specific bar.
      if node.pkg_name == 'com.samsung.android.app.cocktailbarservice':
        layout_utils.delete_branch(node, set_as_bg=keep_all_boxes)

      # Discard internals of Samsung keyboard.
      if (node.pkg_name == 'com.samsung.android.honeyboard' and node.resource_id
          == 'com.samsung.android.honeyboard:id/layout_overlay'):
        for n in node.children:
          layout_utils.delete_branch(n, set_as_bg=keep_all_boxes)

      # Discard internals of Google keyboard.
      if (node.pkg_name == 'com.google.android.inputmethod.latin' and
          node.resource_id
          == 'com.google.android.inputmethod.latin:id/keyboard_holder'):
        for n in node.children:
          layout_utils.delete_branch(n, set_as_bg=keep_all_boxes)

    if options.run_labeling:
      cls = _get_label_name(node=node)

      # Delete top title bar. This is a box at the top of the screen with low
      # height. The NAVIGATION_BAR_ADDITIONAL_LABEL label is dedicated for this
      # purpose.
      is_on_top = bbox.top <= 0.01 and bbox.width >= 0.95 and bbox.height <= 0.05
      if cls == NAVIGATION_BAR_ADDITIONAL_LABEL and is_on_top:
        layout_utils.delete_branch(node, set_as_bg=keep_all_boxes)

      # Correct Radio label.
      if cls == RADIO_LABEL and (bbox.height > 0.8 or bbox.area > 0.4):
        cls = TEXT_BUTTON_LABEL

      # Check for drawer related properties and update arrays if needed.
      if cls == DRAWER_LABEL_NAME and bbox.area > options.drawer_max_area:
        # Invalid drawer - the whole screen cannot be a Drawer element.
        cls = ''

      # Correct wrong sized bounding boxes.
      if cls in LABELS_MEDIUM_TO_SMALL and bbox.area > options.medium_small_max_area:
        metrics.Metrics.counter('semantics', 'box_too_large').inc()
        cls = ''

      if not cls:
        layout_utils.delete_node(node, set_as_bg=keep_all_boxes)
        continue  # Class was not matched.

      if (cls != CONTAINER_ADDITIONAL_LABEL and
          cls != WEB_VIEW_LABEL) and cls not in options.class_to_label:
        # Bad mapping.
        metrics.Metrics.counter('semantics',
                                'PROBLEM: map_missing ' + cls).inc()
        layout_utils.delete_node(node, set_as_bg=keep_all_boxes)
        continue

      # Add webview temporarily.
      if cls == WEB_VIEW_LABEL:
        node.set_labels(cls, -1)  # later either dropped or converted to ad.
      # Add candidate containers.
      elif cls == CONTAINER_ADDITIONAL_LABEL and bbox.area < 0.9:
        node.set_labels(cls, -1)  # later either dropped or CONTAINER.
      elif cls == CONTAINER_ADDITIONAL_LABEL:
        layout_utils.delete_node(node, set_as_bg=keep_all_boxes)
        continue
      else:
        node.set_labels(cls, options.class_to_label[cls])

  # Global (all nodes) analysis.
  # Select container bboxes given candidates.
  if options.run_labeling:
    _select_containers(tree, options.class_to_label, set_as_bg=keep_all_boxes)

    # Relabel images in bottom navigation bar.
    _relabel_images_in_bottom_bar(tree, options.class_to_label)

    # Remove duplicates, boxes behind drawer and overlapping edited webview.
    _clean_webview_adview_bboxes(
        tree,
        options.class_to_label,
        img_aspect_ratio,
        set_as_bg=keep_all_boxes)

    # Remaining web views are filled with white.
    image = layout_utils.empty_bboxes(
        tree, WEB_VIEW_LABEL, example, set_as_bg=keep_all_boxes)

    # Add the root of the tree if it has not been assigned.
    if (tree.bbox.ui_class is None or
        tree.bbox.ui_class == CONTAINER_ADDITIONAL_LABEL):
      tree.set_labels('ROOT', -1)

    # Correct icons, keyboard and text.
    layout_utils.shrink_pager_indicators(tree)
    _clean_text(tree, options.class_to_label, set_as_bg=keep_all_boxes)
    _convert_to_label(tree, options.class_to_label)

  # Remove blank and overlapping boxes, and perform cleaning.
  if options.run_cleaning:
    _remove_boxes_behind_occluder(tree, set_as_bg=keep_all_boxes)
    layout_utils.remove_blank_boxes(tree, image, set_as_bg=keep_all_boxes)

    # Remove empty containers.
    layout_utils.remove_leaf_boxes(tree, CONTAINERS, set_as_bg=keep_all_boxes)

    # Remove overlapping boxes after removing empty containers, because some
    # containers will be empty after the overlapping boxes are removed, and this
    # would remove too many containers for the RICO dataset.
    _remove_overlapping_boxes(tree, set_as_bg=keep_all_boxes)

  # Discard screens with too few boxes.
  if len(tree.descendants) <= 2:
    metrics.Metrics.counter('semantics', 'not_enough_bboxes').inc()
    return None

  # Done, convert back to tf.Example.
  example = tfexample_tree.tree_to_tfexample(tree, example=example)
  return example
