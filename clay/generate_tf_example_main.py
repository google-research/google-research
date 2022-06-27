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

"""Create RICO dataset for layout denoising task."""
import csv
import io
import json
import os

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam import runners
from PIL import Image
import tensorflow as tf

from clay import preprocessing
from clay.proto import observation_pb2
from clay.utils import proto_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('screen_id_file', None, 'File containing RICO screen ids.')
flags.DEFINE_string('input_dir', None,
                    'Folder containing RICO json and jpeg files.')
flags.DEFINE_string('output_path', None,
                    'Output path of the generated tf.Example.')
flags.DEFINE_bool(
    'clean_tf_example', False,
    'Whether to clean tf.Example by preprocessing. The preprocessing is applied'
    ' before labeling the RICO dataset.')
flags.DEFINE_string('csv_label_file', None,
                    'Output path of the generated tf.Example.')

# Feature prefix required for the post-processing script.
_VH_PREFIX = 'image/view_hierarchy'
_VHA_PREFIX = 'image/view_hierarchy/attributes'

LABEL_TO_CLASS = {
    0: 'BACKGROUND',  # Invalid objects.
    1: 'IMAGE',
    2: 'PICTOGRAM',
    3: 'BUTTON',
    4: 'TEXT',
    5: 'LABEL',
    6: 'TEXT_INPUT',
    7: 'MAP',
    8: 'CHECK_BOX',
    9: 'SWITCH',
    10: 'PAGER_INDICATOR',
    11: 'SLIDER',
    12: 'RADIO_BUTTON',
    13: 'SPINNER',
    14: 'PROGRESS_BAR',
    15: 'ADVERTISEMENT',
    16: 'DRAWER',
    17: 'NAVIGATION_BAR',
    18: 'TOOLBAR',
    19: 'LIST_ITEM',
    20: 'CARD_VIEW',
    21: 'CONTAINER',
    22: 'DATE_PICKER',
    23: 'NUMBER_STEPPER',
}

CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}


def fix_json_bbox(json_content, jpg_content, recursive_from_activity_root=True):
  """Fixes bounds in json to align with image."""
  screenshot = Image.open(io.BytesIO(jpg_content))
  width, height = screenshot.size
  w_ratio = width / 1440.
  h_ratio = height / 2560.
  json_dict = json.loads(json_content)

  def limit_x(x):
    return min(max(x, 0), 1440.)

  def limit_y(y):
    return min(max(y, 0), 2560.)

  def fix_box(json_dict):
    if 'bounds' in json_dict:
      x1, y1, x2, y2 = json_dict.get('bounds', [0, 0, 0, 0])
      new_x_y = [
          int(limit_x(x1) * w_ratio),
          int(limit_y(y1) * h_ratio),
          int(limit_x(x2) * w_ratio),
          int(limit_y(y2) * h_ratio)
      ]
      logging.info('Align size: %s -> %s', [x1, y1, x2, y2], new_x_y)
      json_dict['bounds'] = new_x_y

    if 'children' in json_dict:
      for child in json_dict['children']:
        if child:
          fix_box(child)

  if recursive_from_activity_root:
    fix_box(json_dict['activity']['root'])
  else:
    fix_box(json_dict)
  return bytes(json.dumps(json_dict), 'utf-8')


class GenerateProto(beam.DoFn):
  """Generate Proto."""

  def __init__(self, input_dir, clean_tf_example, csv_label_file=None):
    self._input_dir = input_dir
    self._clean_tf_example = clean_tf_example
    self._json_counter = beam.metrics.Metrics.counter(self.__class__, 'json')
    self._missing_json_counter = beam.metrics.Metrics.counter(
        self.__class__, 'missing_json')
    self._missing_image_counter = beam.metrics.Metrics.counter(
        self.__class__, 'missing_image')
    self._no_objects_counter = beam.metrics.Metrics.counter(
        self.__class__, 'no_objects_counter')
    self._fail_counter = beam.metrics.Metrics.counter(self.__class__,
                                                      'fail_parsing_json')
    self._invalid_bbox_counter = beam.metrics.Metrics.counter(
        self.__class__, 'invalid_bbox')
    self._clean_none_counter = beam.metrics.Metrics.counter(
        self.__class__, 'clean_none_example')
    self._output_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'output_example')
    self._missing_label_counter = beam.metrics.Metrics.counter(
        self.__class__, 'node_missing_label')
    # Create cleaning option for preprocessing.
    self._clean_option = preprocessing.CleaningOptions(
        class_to_label=CLASS_TO_LABEL,
        label_to_class=LABEL_TO_CLASS,
        keep_all_boxes=False)

    self._labels = {}
    self._target_screens = set()
    if csv_label_file:
      # Load labels which will be added to tf.Example as features.
      with tf.io.gfile.Gfile(csv_label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
          self._labels[(row['screen_id'], row['node_id'])] = int(row['label'])
          self._target_screens.add(row['screen_id'])

  def process(self, screen_id):

    if self._target_screens and screen_id not in self._target_screens:
      # Only generate examples for the labeled screens, if labels are provided.
      return

    self._json_counter.inc(1)

    json_path = os.path.join(self._input_dir, screen_id + '.json')
    if not tf.io.gfile.exists(json_path):
      # This might happen for the internal NodeRICO dataset.
      self._missing_json_counter.inc(1)
      return

    image_path = os.path.join(self._input_dir, screen_id + '.jpg')
    if not tf.io.gfile.Gfile.exists(image_path):
      self._missing_image_counter.inc(1)
      return

    observation = observation_pb2.Observation()
    observation.debug_vh_filepath = json_path
    observation.debug_screenshot_filepath = image_path

    with tf.io.gfile.Gfile(json_path, 'rb') as f:
      observation.json = f.read()
    with tf.io.gfile.Gfile(image_path, 'rb') as f:
      observation.screenshot = f.read()

    try:
      # In RICO dataset, the bbox in VH is based on [2560x1440], which might be
      # different from the actual image size. Here we fix the bbox with respect
      # to the actual image size.
      observation.json = fix_json_bbox(observation.json, observation.screenshot)
      observation.image_id = os.path.basename(image_path)
      proto_utils.fill_observation_proto(observation, include_invisible=True)
    except:  # pylint: disable=bare-except
      self._fail_counter.inc(1)
      logging.error('Failed to when processing json.')
      return

    if not observation.objects:
      self._no_objects_counter.inc(1)
      return

    example = self.observation_to_tfexample(observation)
    if self._clean_tf_example:
      # This is done before labeling as in the paper.
      example = preprocessing.clean_screen_tfexample(example,
                                                     self._clean_option)

    if not example:
      self._clean_none_counter.inc(1)
      return

    if self._labels:
      f = example.features.feature
      node_ids = f['image/view_hierarchy/node_id'].bytes_list.value
      labels = f['image/object/class/label'].int64_list.value
      new_labels = []
      new_labels_text = []
      for node_id, label in zip(node_ids, labels):
        key = (screen_id, node_id.decode())
        if key not in self._labels:
          self._missing_label_counter.inc(1)
        else:
          label = self._labels[key]
        if label == -1:
          label_text = 'ROOT'
        else:
          label_text = self._id2class[label]
        new_labels_text.append(label_text.encode())
        new_labels.append(label)

      del f['image/object/class/label'].int64_list.value[:]
      del f['image/object/class/text'].bytes_list.value[:]

      f['image/object/class/label'].int64_list.value.extend(new_labels)
      f['image/object/class/text'].bytes_list.value.extend(new_labels_text)

    self._output_counter.inc(1)
    yield screen_id, example

  def observation_to_tfexample(self, proto):
    """Converts observation proto to tf.Example."""
    # Generate tf.Example with features required by layout labeling plugin.
    example = tf.train.Example()
    f = example.features.feature

    screenshot = Image.open(io.BytesIO(proto.screenshot))
    width, height = screenshot.size
    image_bytes = io.BytesIO()
    screenshot.save(image_bytes, screenshot.format)

    f['image/encoded'].bytes_list.value.append(image_bytes.getvalue())
    f['image/height'].int64_list.value.append(height)
    f['image/width'].int64_list.value.append(width)
    f['image/channels'].int64_list.value.append(3)
    f['image/colorspace'].bytes_list.value.append(b'RGB')
    f['image/format'].bytes_list.value.append(screenshot.format.encode())
    f['image/filename'].bytes_list.value.append(
        proto.debug_screenshot_filepath.encode())

    for o in proto.objects:
      x1, x2, y1, y2 = o.bbox.left, o.bbox.right, o.bbox.top, o.bbox.bottom
      if x1 > x2 or y1 > y2:
        self._invalid_bbox_counter.inc(1)
        x1 = min(x1, x2)
        y1 = min(y1, y2)

      f['image/object/bbox/xmin'].float_list.value.append(x1 / width)
      f['image/object/bbox/xmax'].float_list.value.append(x2 / width)
      f['image/object/bbox/ymin'].float_list.value.append(y1 / height)
      f['image/object/bbox/ymax'].float_list.value.append(y2 / height)

      f[f'{_VHA_PREFIX}/checked'].int64_list.value.append(o.checked)
      f[f'{_VHA_PREFIX}/clickable'].int64_list.value.append(o.clickable)
      f[f'{_VHA_PREFIX}/enabled'].int64_list.value.append(o.enabled)
      f[f'{_VHA_PREFIX}/focusable'].int64_list.value.append(o.focusable)
      f[f'{_VHA_PREFIX}/focused'].int64_list.value.append(o.focused)
      f[f'{_VHA_PREFIX}/selected'].int64_list.value.append(o.selected)
      f[f'{_VHA_PREFIX}/visibility'].int64_list.value.append(o.visible)
      f[f'{_VHA_PREFIX}/visible_to_user'].int64_list.value.append(o.visible)
      f[f'{_VHA_PREFIX}/id'].bytes_list.value.append(o.resource_id.encode())

      # Use default values as these features are absent in RICO.
      f[f'{_VHA_PREFIX}/opacity'].int64_list.value.append(1)
      f[f'{_VHA_PREFIX}/elevation'].int64_list.value.append(0)
      f[f'{_VHA_PREFIX}/text_size'].int64_list.value.append(0)

      f[f'{_VH_PREFIX}/node_id'].bytes_list.value.append(o.id.encode())
      f[f'{_VH_PREFIX}/text'].bytes_list.value.append(o.text.encode())
      f[f'{_VH_PREFIX}/description'].bytes_list.value.append(
          o.content_desc.encode())
      f[f'{_VH_PREFIX}/is_leaf'].int64_list.value.append(o.is_leaf)
      f[f'{_VH_PREFIX}/parent_id'].int64_list.value.append(o.parent_index)
      f[f'{_VH_PREFIX}/class/name'].bytes_list.value.append(
          o.android_class.encode())

    return example


def create_pipeline(screen_id_file, input_dir, output_path, clean_tf_example,
                    csv_label_file):
  """Runs the end-to-end beam pipeline."""

  def _pipeline(root):
    _ = (
        root
        | 'Read' >> beam.io.ReadFromText(screen_id_file)
        | 'ReadEpisodeProto' >> beam.ParDo(
            GenerateProto(input_dir, clean_tf_example, csv_label_file))
        | 'ReShuffle' >> beam.Reshuffle()  # workers may not parallel w/o this
        | 'WriteResults' >> beam.io.WriteToTFRecord(
            output_path, coder=beam.coders.ProtoCoder(tf.train.Example)))

  return _pipeline


def main(_):
  pipeline = create_pipeline(FLAGS.screen_id_file, FLAGS.input_dir,
                             FLAGS.output_path, FLAGS.clean_tf_example,
                             FLAGS.csv_label_file)
  runners.DataflowRunner().run_pipeline(pipeline)


if __name__ == '__main__':
  app.run(main)
