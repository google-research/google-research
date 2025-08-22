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

"""Imports CamVid11, the 11 class MSeg version.

URL: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

Paper: G. J. Brostow, J. Fauqueur, and R. Cipolla. Semantic object classes
in video: A high-definition ground truth database. Patt. Rec.
Letters, 2009.
"""
import numpy as np
import tensorflow.compat.v2 as tf

from factors_of_influence.fids import mseg_base
from factors_of_influence.fids import utils


class CamVid(mseg_base.MSegBase):
  """CamVid Dataset with separate dataloader for segmentation masks."""

  def __init__(self):
    super().__init__(
        mseg_name='CamVid',
        mseg_original_name='camvid-32',
        mseg_base_name='camvid-11',
        mseg_dirname='Camvid/',
        mseg_train_dataset=False,
        mseg_segmentation_background_labels=['Void'],
    )
    self._label_defs = None

  def load_label_defs(self):
    """Load color data from colors.txt file."""
    file_dir = f'{mseg_base.MSEG_LABEL_DIR}/{self.mseg_original_name}'
    file_base = f'{file_dir}/{self.mseg_original_name}'
    color_file = f'{file_base}_colors.txt'
    color_string = tf.io.gfile.GFile(color_file, 'r').read()
    color_data = np.fromstring(color_string, dtype=np.uint8, sep='\t')
    color_data = color_data.reshape(-1, 3)

    class_names = utils.load_text_to_list(f'{file_base}_names.txt')

    label_defs = [
        utils.LabelColorDef(name=name, color=color, id=i)
        for i, (name, color) in enumerate(zip(class_names, color_data))
    ]
    return label_defs

  @property
  def label_defs(self):
    if self._label_defs is None:
      self._label_defs = self.load_label_defs()
    return self._label_defs

  def convert_segmentation(self, segmentation):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    # CamVid needs to be converted from rgb color coding to class ids
    segmentation = utils.convert_segmentation_rgb_to_class_id(
        segmentation, self.label_defs)
    return super().convert_segmentation(segmentation)
