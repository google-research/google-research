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

"""Libraries for evaluating visual relationship detection."""

import collections
import enum
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import dataclasses
import numpy
import pandas



# Two visual relationships are considered, i.e. 1) whether one object occludes
# the other, and 2) the relative distance of the two objects to the camera.
class VRDAttribute(enum.Enum):
  OCCLUSION = 'occlusion'
  DISTANCE = 'distance'


@dataclasses.dataclass
class Box:
  """An object represented by its bounding box and entity."""
  image_id: str
  entity: str
  ymin: float
  xmin: float
  ymax: float
  xmax: float


@dataclasses.dataclass
class Record:
  bbox_a: Box
  bbox_b: Box
  occlusion: int
  distance: int


class VRDEvaluator:
  """Evaluator for computing vrd metrics."""

  NUM_LABELS = 4
  TP_IDX = 0
  FP_IDX = 1
  FN_IDX = 2

  def __init__(self,
               groundtruth_path,
               object_path,
               check_entity = False,
               iou_threshold = 0.5):
    """Instantiates vrd evaluator.

    Args:
      groundtruth_path: path to the vrd ground truth.
      object_path: path to the object ground truth.
      check_entity: if True, check the object entity when determining whether an
        object is detected.
      iou_threshold: the threshold for correctly detected objects.
    """
    groundtruths = load_groundtruth(groundtruth_path, object_path)
    self.example_groundtruths = reverse_object_order(groundtruths)
    self.check_entity = check_entity
    self.iou_threshold = iou_threshold

  def is_success_detection(self, prediction, groundtruth):
    """Evaluates whether prediction correctly predicts groundtruth object."""
    if prediction.image_id != groundtruth.image_id:
      return False
    if self.check_entity and prediction.entity != groundtruth.entity:
      return False
    iou = compute_iou(prediction, groundtruth)
    return iou > self.iou_threshold

  def is_correct_prediction(
      self,
      prediction,
      groundtruth,
      attr = VRDAttribute.DISTANCE):
    """Evaluates whether prediction correctly predicts groundtruth vrd record.

    A prediction predicts a groundtruth vrd record if 1) both obj_a and obj_b
    in the gorund truth are correctly detected by obj_a and obj_b in the
    prediction respectively, and 2) the relationship between obj_a and obj_b
    are predicted correctly.

    Args:
      prediction: predicted vrd triplet.
      groundtruth: ground truth vrd triplet.
      attr: the visual relationship to be considered.

    Returns:
      Whether the prediction correctly predicts the ground truth.
    """
    if not self.is_success_detection(prediction.bbox_a, groundtruth.bbox_a):
      return False
    if not self.is_success_detection(prediction.bbox_b, groundtruth.bbox_b):
      return False
    return getattr(prediction, attr.value) == getattr(groundtruth, attr.value)

  def evaluate_example(
      self,
      predictions,
      groundtruths,
      attr = VRDAttribute.DISTANCE):
    """Evaluates the vrd results for one image pair.

    Computes the number of true positives, false positives, and false negatives
    for the vrd results on one image pair. A predicted vrd triplet is considered
    as true positive if 1) it correctly predicts a ground truth vrd triplet, and
    2) the ground truth vrd triplet is not predicted by other predictions.
    Following object detection metrics, duplicate detections will be considered
    as false positives. Please refer to ``is_correct_prediction'' for
    more information about condition 1). Note that this function does not
    consider the vrd triplets in reversed order, i.e.
    (obj_a, obj_b, relationship) will not be translated to
    (obj_b, obj_a, ~relationship) automatically.

    Args:
      predictions: predicted vrd (obj_a, obj_b, relationship) triplets.
      groundtruths: ground truth vrd (obj_a, obj_b, relationship) triplets.
      attr: the relationship to be evaluated.

    Returns:
      A [4, 3] numpy.ndarray. Each row corresponds to a possible label for the
      relationship, and the columns correspond to true positives, false
      positives, and false negatives respectively.
    """
    # detected[i] indicates whether groundtruths[i] has been correctly
    # detected by one of the predictions. Following the metric definition of
    # object detection, duplicate detections will be considered as false
    # positives.
    detected = [False] * len(groundtruths)

    def match_groundtruth(prediction, detected):
      # Checks if the prediction is a true or false positive.
      for i, groundtruth in enumerate(groundtruths):
        if detected[i]:
          continue
        if self.is_correct_prediction(prediction, groundtruth, attr=attr):
          return i, detected
      return -1, detected

    results = numpy.zeros((self.NUM_LABELS, 3))
    for prediction in predictions:
      label = getattr(prediction, attr.value)
      i, detected = match_groundtruth(prediction, detected)
      if i < 0:
        results[label, self.FP_IDX] += 1
      else:
        detected[i] = True
        results[label, self.TP_IDX] += 1
    for i, groundtruth in enumerate(groundtruths):
      if not detected[i]:
        label = getattr(groundtruth, attr.value)
        results[label, self.FN_IDX] += 1
    return results

  # TODO(ycsu): update the paper link after arxiv version is ready.
  def compute_metrics(
      self,
      example_predictions,
      filter_fn = None,
  ):
    """Computes vrd metrics for example_predictions.

    Computes the metrics for visual relationship detection. Note that the
    evaluator assumes the relationship is predicted from both directions for
    each pair of objects, i.e. both (obj_a, obj_b, relationship) and
    (obj_b, obj_a, relationship) for each pair of (obj_a, obj_b).

    Args:
      example_predictions: a mapping between an image pair and all visual
        relationshp triplets detected from the image pair.
      filter_fn: a function that takes a VRD record as input and determines
        whether the record should be considered during evaluation. This is
        used to analyze VRD performance based on object properties.

    Returns:
      A pandas.DataFrame that has 10 rows and the following columns:
        'relationship': the relationship {occlusion,distance} being evaluated.
        'label': the label {0, 1, 2, 3} for the relationship, or 'all' for the
          results of all labels jointly.
        'precision': the precision value.
        'recall': the recall value.
        'fscore': the f1score value.
    """
    accumulated = {
        attr: numpy.zeros((self.NUM_LABELS, 3)) for attr in VRDAttribute
    }

    for example_id, groundtruths in self.example_groundtruths.items():
      predictions = example_predictions.get(example_id, [])
      if filter_fn is not None:
        groundtruths = [record for record in groundtruths if filter_fn(record)]
        predictions = [record for record in predictions if filter_fn(record)]
      for attr in VRDAttribute:
        result = self.evaluate_example(
            predictions, groundtruths, attr=attr)
        accumulated[attr] += result

    dataframe = []
    for attr, results in accumulated.items():
      for label in range(results.shape[0]):
        metrics = compute_metrics(*results[label])
        dataframe.append([attr.value, label, *metrics.values()])
      metrics = compute_metrics(*results.sum(axis=0))
      dataframe.append([attr.value, 'all', *metrics.values()])
    return pandas.DataFrame(
        dataframe,
        columns=['relationship', 'label', 'precision', 'recall', 'fscore'])


def reverse_object_order(
    example_records
):
  """Creates new records by reversing the object order in existing records.

  A visual relationship is defined by (obj_a, obj_b, relationship) where the
  relationship is directional. For each record in the input, this function
  creates a new triplet by changing the order of (obj_a, obj_b) and the
  relationship label accordingly.  The new records are inserted into the input
  example_records, so the returned dictionary contains both
  (obj_a, obj_b, relationship) and (obj_b, obj_a, ~relationship) for each
  (obj_a, obj_b) pair.

  Args:
    example_records: a mapping between image pair and VRD triplets.

  Returns:
    A dictionary that maps an image pair to the VRD triplets with both the
    original and reversed order.
  """
  reversed_records = collections.defaultdict(list)
  for image_pair, records in example_records.items():
    reversed_records[image_pair].extend(records)
    for record in records:
      reversed_record = Record(record.bbox_b, record.bbox_a, 0, 0)
      for attr in VRDAttribute:
        original_label = getattr(record, attr.value)
        # Change the label value after swapping the object order if the label is
        # directional, e.g. A is closer than B -> B is closer than B. Do nothing
        # if the label is non-directional, e.g. A and B are about the same
        # distance.
        if original_label == 1:
          reversed_label = 2
        elif original_label == 2:
          reversed_label = 1
        else:
          reversed_label = original_label
        setattr(reversed_record, attr.value, reversed_label)
      reversed_records[image_pair].append(reversed_record)
  return reversed_records


def compute_area(bbox):
  """Computes the normalized bounding box area."""
  return max(0., bbox.xmax - bbox.xmin) * max(0., bbox.ymax - bbox.ymin)


def compute_iou(bbox_a, bbox_b):
  """Computes intersection over union between two bounding boxes."""
  xmin = max(bbox_a.xmin, bbox_b.xmin)
  xmax = min(bbox_a.xmax, bbox_b.xmax)
  ymin = max(bbox_a.ymin, bbox_b.ymin)
  ymax = min(bbox_a.ymax, bbox_b.ymax)
  overlap = Box(bbox_a.image_id, 'overlap', ymin, xmin, ymax, xmax)
  area_overlap = compute_area(overlap)
  area_a = compute_area(bbox_a)
  area_b = compute_area(bbox_b)
  return area_overlap / (area_a + area_b - area_overlap)


def compute_metrics(tp, fp, fn):
  """Computes precision, recall, and fscore."""
  if tp < 0:
    raise ValueError('True positive should be non-negative.')
  if fp < 0:
    raise ValueError('False positive should be non-negative.')
  if fn < 0:
    raise ValueError('False negative should be non-negative.')
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = 0.
  if tp + fn > 0.:
    recall = tp / (tp + fn)
  else:
    recall = 0.

  if precision + recall > 0.:
    fscore = 2. * precision * recall /  (precision + recall)
  else:
    fscore = 0.
  return dict(precision=precision, recall=recall, fscore=fscore)


def convert_dataframe_to_records(
    dataframe):
  """Converts pandas dataframe to vrd records.

  Converts each row in the dataframe to a vrd record. Each record consists of a
  (object_a, object_b, visual relationship) triplet, and each object is defined
  by the image_id, a 2D bounding box in the image, and the object entity. The
  visual relationship considered in this codebase is the relative distance and
  occlusion relationship. The records will be grouped by the image_id of the two
  objects before output.

  Args:
    dataframe: a pandas.DataFrame with the following columns: image_id_{1,2},
      entity_{1,2}, xmin_{1,2}, xmax_{1,2}, ymin_{1,2}, ymax_{1,2}, occlusion,
      distance.

  Returns:
    A dictionary that maps an image pair to the detected
    (object, object, relationship) triplets within the two images.
  """
  records = collections.defaultdict(list)
  for _, row in dataframe.iterrows():
    bbox_a = Box(row.image_id_1, row.entity_1, row.ymin_1, row.xmin_1,
                 row.ymax_1, row.xmax_1)
    bbox_b = Box(row.image_id_2, row.entity_2, row.ymin_2, row.xmin_2,
                 row.ymax_2, row.xmax_2)
    record = Record(bbox_a, bbox_b, row.occlusion, row.distance)
    records[(row.image_id_1, row.image_id_2)].append(record)
  return records


def load_groundtruth(
    groundtruth_path,
    object_path):
  """Loads the ground truths for visual relationship detection.

  This function loads the visual relationship ground truths from csv files.
  Please refer to the dataset description of ``2.5D Visual Relationship
  Detection'' for more information about the dataset and data format.

  Args:
    groundtruth_path: path to the vrd ground truth.
    object_path: path to the object ground truth.

  Returns:
    A dictionary that maps an image pair to the detected
    (object, object, relationship) triplets within the two images.
  """
  with open(groundtruth_path, 'r') as groundtruth_file:
    groundtruth = pandas.read_csv(groundtruth_file)

  with open(object_path, 'r') as object_file:
    objects = pandas.read_csv(object_file)

  groundtruth = groundtruth.merge(
      objects,
      left_on=['image_id_1', 'object_id_1'],
      right_on=['image_id', 'object_id'],
  ).drop(
      ['image_id', 'object_id'],
      axis=1,
  ).merge(
      objects,
      left_on=['image_id_2', 'object_id_2'],
      right_on=['image_id', 'object_id'],
      suffixes=['_1', '_2'],
  ).drop(
      ['image_id', 'object_id'],
      axis=1,
  )

  return convert_dataframe_to_records(groundtruth)


def load_prediction(
    prediction_path):
  """Loads visual relationship detection results.

  The visual relationship detection results should be stored in a csv file,
  where each row corresponds to one detection, i.e. a pair of objects and the
  predicted visual relationships. The csv file should contain the following
  columns:
    image_id_{1,2}: the image_id for the {first,second} object.
    entity_id_{1,2}: the entity id for the {first,second} object.
    xmin_{1,2}, xmax_{1,2}, ymin_{1,2}, ymax_{1,2}: the bounding box coordinates
      for the {first,second} object. The coordinates should be normalized to
      [0, 1].
    occlusion: the predicted occlusion relationship between the two objects.
      Possible values and their meanings are:
        0: No occlusion
        1: The first object occludes the second object
        2: The second object occludes the first object
        3: Mutually occluded
    distance: the predicted relative distance relationship between the two
      objects. Possible values and their meanings are:
        0: Unsure
        1: The first object is closer to the camera
        2: The second object is closer to the camera
        3: The two objects are about the same distance
  Please refer to the paper for more information about the visual relationship
  definition.

  Args:
    prediction_path: the path to the prediction result.

  Returns:
    A dictionary that maps an image pair to the detected
    (object, object, relationship) triplets within the two images.
  """
  with open(prediction_path, 'r') as prediction_file:
    predictions = pandas.read_csv(prediction_file)

  return convert_dataframe_to_records(predictions)


def get_filter_boundingbox_fn(rng_a,
                              rng_b,
                              bbox_attr):
  """Creates a filter function for VRD record based on bounding box attributes.

  The filter function determines if the attributes of the object bounding
  boxes in a VRD records is in [rng[0], rng[1]). The record should be kepted
  after filtering if both objects satisfies the condition defined by bbox_attr
  and rng_{a,b}.

  Args:
    rng_a: the target range [lower, upper) of the attribute for obj_a.
    rng_b: the target range [lower, upper) of the attribute for obj_b.
    bbox_attr: the target bounding box attribute. Possible vlaues include
      'size', 'horizontal_position', and 'vertical_position'.

  Returns:
    A filter function based on object bounding box attribute.
  """

  def filter_fn(record):
    boxes = [record.bbox_a, record.bbox_b]
    rngs = [rng_a, rng_b]
    for box, rng in zip(boxes, rngs):
      if bbox_attr == 'size':
        value = compute_area(box)
      elif bbox_attr == 'horizontal_position':
        value = (box.xmin + box.xmax) / 2.
      elif bbox_attr == 'vertical_position':
        value = (box.ymin + box.ymax) / 2.
      else:
        raise NotImplementedError(
            f'Unknown bounding box attribute {bbox_attr}.')
      if value < rng[0] or value >= rng[1]:
        return False
    return True

  return filter_fn
