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

# Lint as: python3
"""Bridge Contrastive Learning Heads."""

import tensorflow as tf

from vatt.modeling.heads import mlp_lib


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class FACHead(tf.keras.layers.Layer):
  """MLP-based Head to bridge audio, text and video with a FAC style."""

  def __init__(
      self,
      bn_config,
      use_xreplica_bn,
      vid_to_aud_txt_kwargs,
      aud_to_vid_txt_kwargs,
      txt_to_vid_aud_kwargs,
      name="mlp_fac_head",
      **kwargs):
    """Initialize the Fine-to-Coarse head class.

    Args:
      bn_config: batchnorm configuration args
      use_xreplica_bn: whether to use cross-replica bn stats or not
      vid_to_aud_txt_kwargs: vid2rest MLP args
      aud_to_vid_txt_kwargs: aud2rest MLP args
      txt_to_vid_aud_kwargs: txt2rest MLP args
      name: graph name.
      **kwargs: additional args
    """
    super(FACHead, self).__init__(name=name)
    # vid-to-va is Dense + BN + Relu + Dense + BN
    self.vid_to_hid = tf.keras.layers.Dense(vid_to_aud_txt_kwargs["d_model"],
                                            use_bias=False,
                                            name="vid_to_hid")
    self.hid_to_va = mlp_lib.ReluDenseBN(
        pre_bn=True,
        d_model=vid_to_aud_txt_kwargs["d_model"],
        bn_config=bn_config,
        use_xreplica_bn=use_xreplica_bn,
        name="hid_to_va",
        )

    # aud-to-va is Dense
    self.aud_to_va = tf.keras.layers.Dense(aud_to_vid_txt_kwargs["d_model"],
                                           name="aud_to_vid")

    # va-to-vat is Relue + Dense + BN
    self.va_to_vat = mlp_lib.ReluDenseBN(
        d_model=txt_to_vid_aud_kwargs["d_model"],
        bn_config=bn_config,
        use_xreplica_bn=use_xreplica_bn,
        name="va_to_vat",
        )

    # txt-to-vat is Dense
    self.txt_to_vat = tf.keras.layers.Dense(txt_to_vid_aud_kwargs["d_model"],
                                            name="txt_to_vid")

  def call(self,
           inputs,
           training=False):
    """Computes video, text and audio embeddings.

    Args:
      inputs:
      training:

    Returns:
      outputs:
    """
    video_representation = inputs["video"]["features_pooled"]
    audio_representation = inputs["audio"]["features_pooled"]
    text_representation = inputs["text"]["features_pooled"]

    # mapping of video to va space
    vid2hid = self.vid_to_hid(inputs=video_representation,
                              training=training)
    vid2aud = self.hid_to_va(inputs=vid2hid,
                             training=training)

    # mapping of vid2aud to vat space
    vid2txt = self.va_to_vat(inputs=vid2aud,
                             training=training)

    # mapping of audio to va space
    aud2vid = self.aud_to_va(inputs=audio_representation,
                             training=training)

    # mapping of text to vat space
    txt2vid = self.txt_to_vat(inputs=text_representation,
                              training=training)

    video_embd = {"toaud": vid2aud,
                  "totxt": vid2txt}

    audio_embd = {"tovid": aud2vid}

    text_embd = {"tovid": txt2vid}

    outputs = {
        "video": video_embd,
        "audio": audio_embd,
        "text": text_embd
    }

    return outputs


class JointHead(tf.keras.layers.Layer):
  """MLP-based Head to bridge audio, text and video with a Joint style."""

  def __init__(
      self,
      bn_config,
      use_xreplica_bn,
      vid_to_aud_txt_kwargs,
      aud_to_vid_txt_kwargs,
      txt_to_vid_aud_kwargs,
      name="mlp_fac_head",
      **kwargs):
    """Initialize the Fine-to-Coarse head class.

    Args:
      bn_config: batchnorm configuration args
      use_xreplica_bn: whether to use cross-replica bn stats or not
      vid_to_aud_txt_kwargs: vid2rest MLP args
      aud_to_vid_txt_kwargs: aud2rest MLP args
      txt_to_vid_aud_kwargs: txt2rest MLP args
      name: graph name.
      **kwargs: additional args
    """
    super(JointHead, self).__init__(name=name)
    assert all(
        [vid_to_aud_txt_kwargs["d_model"] == aud_to_vid_txt_kwargs["d_model"],
         vid_to_aud_txt_kwargs["d_model"] == txt_to_vid_aud_kwargs["d_model"]]
        ), "The joint space projection should be the same for all projections"
    d_joint = vid_to_aud_txt_kwargs["d_model"]
    # vid-to-vat is Relu + Dense + BN
    self.vid_to_vat = mlp_lib.ReluDenseBN(
        pre_bn=False,
        d_model=d_joint,
        bn_config=bn_config,
        use_xreplica_bn=use_xreplica_bn,
        name="vid_to_vat",
        )

    # aud-to-vat is Relu + Dense + BN
    self.aud_to_vat = mlp_lib.ReluDenseBN(
        pre_bn=False,
        d_model=d_joint,
        bn_config=bn_config,
        use_xreplica_bn=use_xreplica_bn,
        name="aud_to_vat",
        )

    # txt-to-vat is Relu + Dense + BN
    self.txt_to_vat = mlp_lib.ReluDenseBN(
        pre_bn=False,
        d_model=d_joint,
        bn_config=bn_config,
        use_xreplica_bn=use_xreplica_bn,
        name="txt_to_vat",
        )

  def call(self,
           inputs,
           training=False):
    """Computes video, text and audio embeddings.

    Args:
      inputs:
      training:

    Returns:
      outputs:
    """
    video_representation = inputs["video"]["features_pooled"]
    audio_representation = inputs["audio"]["features_pooled"]
    text_representation = inputs["text"]["features_pooled"]

    # mapping of video to vat space
    vid2vat = self.vid_to_vat(inputs=video_representation,
                              training=training)

    # mapping of audio to vat space
    aud2vat = self.aud_to_vat(inputs=audio_representation,
                              training=training)

    # mapping of text to vat space
    txt2vat = self.txt_to_vat(inputs=text_representation,
                              training=training)

    video_embd = {"toaud": vid2vat,
                  "totxt": vid2vat}

    audio_embd = {"tovid": aud2vat}

    text_embd = {"tovid": txt2vat}

    outputs = {
        "video": video_embd,
        "audio": audio_embd,
        "text": text_embd
    }

    return outputs
