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

"""Hanasu Python support utilities."""

import os.path

from google.protobuf import descriptor
from google.protobuf import text_format
from lingvo import compat as tf
from lingvo.core import hyperparams


class Error(Exception):
  """Run-time error in util."""


class Util:
  """Generate default hyper params for dynamic config components."""

  @classmethod
  def CreateParamsForMessage(cls, m, omit=None):
    """Proto to hyper params conversion.

    Method will define hyperparameters matching a protocol message.  Each field
    added will match the proto field names. Nested messages will be added
    recursively. Parameters will be initialized to None or the empty list.
    Note that one_of messages are skipped and no hyper parameters are added
    for those fields.

    Args:
      m: a protocol message
      omit: an iterable of field names to omit in adding as hyper parameter.

    Returns:
      A hyper parameter object reflecting the protocol message.
    """
    p = hyperparams.Params()
    if not m:
      return p
    desc = "See comments for {msg} for description".format(msg=m.full_name)
    for f in m.fields:
      if f.containing_oneof:
        continue
      if omit and f.full_name in omit:
        continue
      if f.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        p.Define(f.name, [], desc)
      elif f.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        p.Define(f.name, Util.CreateParamsForMessage(f.message_type, omit=omit),
                 desc)
      else:
        p.Define(f.name, None, desc)
    return p

  @classmethod
  def CreateMessageForParams(cls, hparams, msg):
    """Hyper params to proto conversion.

    Populate the proto msg with the hparams values. If a named hyper parameter
    is not in the descriptor of the message it is ignored. Nested hyper params
    populate nested messages. Hyper parameter lists populate repeated message
    fields.

    Args:
      hparams: hyper params to convert for
      msg: protocol message to populate
    """
    msg_descriptor = msg.DESCRIPTOR
    for (key, val) in hparams.IterParams():
      if key not in msg_descriptor.fields_by_name:
        continue
      field_descriptor = msg_descriptor.fields_by_name[key]
      if field_descriptor.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        if (field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED
           ):
          for nested_hparams in val:
            Util.CreateMessageForParams(nested_hparams, getattr(msg, key).add())
        else:
          Util.CreateMessageForParams(val, getattr(msg, key))
      else:
        if (field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED
           ):
          if val is not None:
            getattr(msg, key).extend(val)
        else:
          if val is not None:
            setattr(msg, key, val)

  @classmethod
  def TextParseMessageFromFileWithSearchPaths(cls, fn, search_paths, msg):
    """Parse msg from fn on any search_path."""
    if os.path.isabs(fn):
      try:
        with tf.io.gfile.GFile(fn, mode="r") as f:
          text_format.Parse(f.read(), msg)
      except:  # pylint: disable=bare-except
        tf.logging.error("Failed to parse pipeline from " + fn)
    else:
      for path in search_paths:
        ffn = os.path.join(path, fn)
        try:
          with tf.io.gfile.GFile(ffn, mode="r") as f:
            text_format.Parse(f.read(), msg)
        except:  # pylint: disable=bare-except
          continue
        else:
          break
      else:
        tf.logging.error(
            "Did not find a pipeline on any search path for " + fn)
