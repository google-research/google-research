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

"""Main model file."""

import dataclasses
import functools
import itertools
import math
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple, TypeVar

from absl import logging
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers

from vct.src import elic
from vct.src import entropy_model
from vct.src import metric_collection
from vct.src import schedule
from vct.src import tf_memoize
from vct.src import video_tensors


Metrics = metric_collection.Metrics
WithMetrics = metric_collection.WithMetrics
Tensor = tf.Tensor


def is_iframe(frame_index):
  return frame_index == 0


class Bottleneck(NamedTuple):
  """Bottleneck representing a single frame.."""

  latent_q: tf.Tensor
  bits: tf.Tensor  # (B,)

  entropy_model_features: Optional[tf.Tensor] = None


# Previous latents quantized, as well previous latents quantized and also
# processed by the entropy model.
State = Tuple[entropy_model.PreviousLatent, Ellipsis]

# We can have None if the model does not produce reconstructions.
Image = tf.Tensor

# Hardcoded schedules, in ratios (of steps) or of base values.
HIGHER_LAMBDA_UNTIL = 0.15
HIGHER_LAMBDA_FACTOR = 10.0


class CompressionSchedule(schedule.KerasSchedule):
  """LR Schedule for compression, with a drop at the end and warmup."""

  def __init__(
      self,
      base_learning_rate,
      num_steps,
      warmup_until = 0.02,
      drop_after = 0.85,
      drop_factor = 0.1,
  ):
    super().__init__(
        base_value=base_learning_rate,
        warmup_steps=int(warmup_until * num_steps),
        vals=[1., drop_factor],
        boundaries=[int(drop_after * num_steps)])


LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


class _WrapAsWeightDecaySchedule(LearningRateSchedule):
  """Wraps a learning rate schedule into a weight decay schedule."""

  # This class is needed because we want to multiply the weight decay factor
  # for AdamW with the learning rate schedule. This only works for
  # tfa_optimizers.AdamW if we have a subclass of `LearningRateSchedule`.

  def __init__(self, lr_schedule, weight_decay):
    super().__init__()
    self._lr_schedule = lr_schedule
    self._weight_decay = weight_decay

  def __call__(self, step):
    return self._weight_decay * self._lr_schedule(step)


Cls = Callable[Ellipsis, tf.Module]


def build(cls, **kwargs):
  """Instantiates `cls` with `kwargs."""
  return cls(**kwargs)


@dataclasses.dataclass
class TransformConfig:
  """Configuration for {analysis,synthesis}_{image,glow,residual}.

  Attributes:
    num_channels: Number of channels in the latent space. Used through the arch.
    analysis_image: Transform to latent space from image space.
    synthesis_image: Transform from latent space to image space.
    vct_entropy_model_kwargs: Optional kwargs for VCTEntropyModel.
  """

  num_channels: int = 192

  analysis_image: Cls = functools.partial(
      elic.ElicAnalysis, channels=(128, 160, 192, 192))
  synthesis_image: Cls = functools.partial(
      elic.ElicSynthesis, channels=(192, 160, 128, 3))

  vct_entropy_model_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=dict)

  @classmethod
  def light_transforms(cls):
    """Returns a light-weight config for testing."""
    return TransformConfig(
        num_channels=8,
        analysis_image=functools.partial(
            elic.ElicAnalysis, channels=(8, 8, 8, 8)),
        synthesis_image=functools.partial(
            elic.ElicSynthesis, channels=(8, 8, 8, 3)),
        vct_entropy_model_kwargs=dict(
            window_size_enc=4,
            window_size_dec=2,
            d_model=8,
            num_head=2,
            num_layers_encoder_sep=1,
            num_layers_decoder=2,
            mlp_expansion=2,
        ),
    )


class EncodeOut(NamedTuple):
  bottleneck: Bottleneck
  metrics: Metrics


class NetworkOut(NamedTuple):
  reconstruction: Image
  bits: tf.Tensor
  frame_metrics: Metrics


def _make_optimizer_and_lr_schedule(
    schedules_num_steps,
    weight_decay = 0.03,
    learning_rate = 1e-4,
    global_clipnorm = 1.0,
):
  """Returns optimizer and learning rate schedule."""

  lr_schedule = CompressionSchedule(
      base_learning_rate=learning_rate,
      num_steps=schedules_num_steps,
  )
  opt = tfa_optimizers.AdamW(
      learning_rate=lr_schedule,
      # NOTE: We only implement the weight-decay variant where the factor
      # multiplies the LR schedule.
      # Need an instance of LearningRateSchedule, hence the wrap!
      weight_decay=_WrapAsWeightDecaySchedule(lr_schedule, weight_decay),
      global_clipnorm=global_clipnorm,
      beta_1=0.9,
      beta_2=0.98,
      epsilon=1e-9,
  )
  return opt, lr_schedule


class ResidualBlock(tf.keras.layers.Layer):
  """Standard residual block."""

  def __init__(self, filters, kernel_size, activation = "relu"):
    super().__init__()
    self._conv_1 = tf.keras.layers.Conv2D(
        filters, kernel_size, padding="same")
    self._act_1 = tf.keras.layers.LeakyReLU()
    self._conv_2 = tf.keras.layers.Conv2D(
        filters, kernel_size, padding="same")
    self._act_2 = tf.keras.layers.LeakyReLU()

  def call(self, inputs):
    filtered_1 = self._act_1(self._conv_1(inputs))
    filtered_2 = self._act_2(self._conv_2(filtered_1))
    return filtered_2 + inputs


class Dequantizer(tf.Module):
  """Implements dequantization.

  We feed y' = y + f(z) to the synthesis transform,
  where y is the latent and z is transformer/entropy model features.
  """

  def __init__(self,
               num_channels,
               d_model,
               name = "Dequantizer"):
    """Instantiates dequantizer."""
    super().__init__(name=name)
    self._d_model = d_model
    self._num_channels = num_channels
    with self.name_scope:
      self._process = tf.keras.Sequential([
          tf.keras.layers.Dense(num_channels),
          tf.keras.layers.LeakyReLU(),
          ResidualBlock(
              num_channels, kernel_size=3, activation="lrelu"),
      ])

  @tf.Module.with_name_scope
  def __call__(self, *, latent_q,
               entropy_features):
    """Calculates y'."""
    if entropy_features is None:
      b, h, w, _ = latent_q.shape
      entropy_features = tf.zeros((b, h, w, self._d_model), dtype=tf.float32)
    return latent_q + self._process(entropy_features)


class PerChannelWeight(tf.Module):
  """Learns a weight per channel and broadcasts these.

  This is used to get fake previous frames to encode the first frame.
  """

  def __init__(self,
               num_channels,
               name = None):
    super().__init__(name=name)
    self.weight = tf.Variable(
        tf.random.uniform((num_channels,)), trainable=True, name="weight")

  def __call__(self, latent_shape):
    assert latent_shape[-1] == self.weight.shape[0]
    return tf.ones(latent_shape, self.weight.dtype) * self.weight


def pad_tensor(
    tensor,
    target_factor,
    mode = "REFLECT"):
  """Pad `tensor` to be divisible by `target_factor`."""
  if target_factor <= 0:
    raise ValueError("target_factor must be positive (not %d)" % target_factor)
  if target_factor == 1:  # never need to pad in this case
    return tensor
  _, height, width, _ = tensor.shape
  height_padded = math.ceil(height / target_factor) * target_factor
  width_padded = math.ceil(width / target_factor) * target_factor
  return tf.pad(tensor, [[0, 0], [0, height_padded - height],
                         [0, width_padded - width], [0, 0]], mode)


# Represents tensors, tuple of tensors, dicts of tensors, etc.
_TensorStructure = TypeVar("_TensorStructure")


def _round_if_not_training(tensor, training):
  if not training:
    return tf.round(tensor)
  else:
    return tensor


def _mse_psnr(original, reconstruction,
              training):
  """Calculates mse and PSNR.

  If training is False, we quantize the pixel values before calculating the
  metrics.

  Args:
    original: Image, in [0, 1].
    reconstruction: Reconstruction, in [0, 1].
    training: Whether we are in training mode.

  Returns:
    Tuple mse, psnr.
  """
  # The images/reconstructions are in [0...1] range, but we scale them to
  # [0...255] before computing the MSE.
  mse_per_batch = tf.reduce_mean(
      tf.math.squared_difference(
          _round_if_not_training(original * 255.0, training),
          _round_if_not_training(reconstruction * 255.0, training)),
      axis=(1, 2, 3))
  mse = tf.reduce_mean(mse_per_batch)
  psnr_factor = -10. / tf.math.log(10.)
  psnr = tf.reduce_mean(psnr_factor * tf.math.log(mse_per_batch / (255.**2)))
  return mse, psnr


def _iter_padded(
    tensor_structures,
    target_factor,
    mode = "REFLECT",
):
  """Pad each tensor structure such that it is divisible by `target_factor`."""
  pad = functools.partial(
      pad_tensor, target_factor=target_factor, mode=mode)
  for tensor_structure in tensor_structures:
    yield tf.nest.map_structure(pad, tensor_structure)


def _spy_spatial_shape(
    frames
):
  """Return (height, width) from the first frame, and also the full iterator."""
  it = iter(frames)
  head = next(it)
  spatial_shape = head.spatial_shape
  return spatial_shape, itertools.chain([head], it)


class Model(tf.Module):
  """Encapsulates model + loss + optimizer + metrics.

  Attributes:
    global_step: A (non-trainable) tf.Variable containing the global step.
  """

  def __init__(
      self,
      schedules_num_steps = 750_000,
      rd_lambda = 0.01,
      context_len = 2,
      range_code_transformer = False,
      lightweight = False
  ):
    """Initializes the model.

    Args:
      schedules_num_steps: The total number of training steps (used for
        schedules).
      rd_lambda: The rate-distortion trade-off parameter (weighing the
        distortion).
      context_len: How many previous latents to feed.
      range_code_transformer: Use for eval to enable range coding (i.e.,
        calculatate bitrates using entropy coding)
      lightweight: If given, use lightweight transforms.
    """
    super().__init__()
    self._all_trainable_variables = None
    self._did_jit_for_eval = False
    self._context_len = context_len
    self._schedules_num_steps = schedules_num_steps
    self._rd_lambda = rd_lambda
    self._pad_factor = 16
    self._optimizer, self._learning_rate_schedule = (
        _make_optimizer_and_lr_schedule(schedules_num_steps))
    if lightweight:
      self._transform_config = TransformConfig.light_transforms()
    else:
      self._transform_config = TransformConfig()
    self._range_code_transformer = range_code_transformer
    self._temporal_pad_token_maker = PerChannelWeight(
        num_channels=self._transform_config.num_channels)
    self._build_transforms(self._transform_config)

  def _build_transforms(self, config):
    logging.info("Building analysis_image=%s", config.analysis_image)
    logging.info("Building synthesis_image=%s", config.synthesis_image)
    logging.info("Building VCT with kargs=%s", config.vct_entropy_model_kwargs)

    self._analysis_image = build(
        config.analysis_image, output_channels=config.num_channels)
    self._synthesis_image = build(config.synthesis_image, output_channels=3)
    self._entropy_model_pframe = entropy_model.VCTEntropyModel(
        num_channels=config.num_channels, **config.vct_entropy_model_kwargs)

    self._dequantizer = Dequantizer(config.num_channels,
                                    self._entropy_model_pframe.d_model)

  @property
  def global_step(self):
    """Returns the global step variable."""
    return self._optimizer.iterations

  def _scheduled_rd_lambda(self,
                           index = None):
    """Returns the scheduled rd-lambda."""
    schedule_value = schedule.schedule_at_step(
        self.global_step,
        vals=[HIGHER_LAMBDA_FACTOR, 1.],
        boundaries=[int(self._schedules_num_steps * HIGHER_LAMBDA_UNTIL)],
    )
    return tf.convert_to_tensor(self._rd_lambda) * schedule_value

  def _encode_iframe_latent_with_pframe_model(
      self,
      latent,
      training,
  ):
    fake_previous_latent = self._temporal_pad_token_maker(latent.shape)
    assert fake_previous_latent.shape == latent.shape  # Programmer error.
    processed = (
        self._entropy_model_pframe.process_previous_latent_q(
            fake_previous_latent, training=training))
    return self._entropy_model_pframe(
        latent_unquantized=latent,
        previous_latents=(processed,),
        training=training)

  def _encode_iframe_latent(
      self,
      latent,
      training,
  ):
    """Encodes the I-frame latent."""
    return self._encode_iframe_latent_with_pframe_model(latent, training)

  def encode_iframe(
      self,
      frame,
      training,
      cache,
  ):
    latent = self._analysis_image(frame.rgb, training=training)
    output = self._encode_iframe_latent(latent, training)
    metrics = output.metrics
    bottleneck = Bottleneck(output.perturbed_latent,
                            output.bits, output.features)
    decode_iframe = tf_memoize.bind(self.decode_iframe, cache)
    _, state, _ = decode_iframe(bottleneck, training)
    return state, EncodeOut(bottleneck, metrics)

  @tf_memoize.memoize
  def decode_iframe(
      self,
      bottleneck,
      training,
  ):
    metrics = Metrics.make()
    latent_q = bottleneck.latent_q
    synthesis_in = self._dequantizer(
        latent_q=latent_q, entropy_features=bottleneck.entropy_model_features)
    reconstruction = self._synthesis_image(synthesis_in, training=training)
    latent_q = bottleneck.latent_q
    # TODO(mentzer): Remove.
    latent_q = tf.stop_gradient(latent_q)
    previous_latent = self._entropy_model_pframe.process_previous_latent_q(
        latent_q, training=training)
    # Note that this is a tuple, we start with a 1-length context.
    state: State = (previous_latent,)
    return reconstruction, state, metrics

  def encode_pframe(
      self,
      frame,
      frame_index,
      state,  # \hat y_t-1
      training,
      cache,
      ):
    metrics = Metrics.make()
    latent = self._analysis_image(frame.rgb, training=training)

    if not training and self._range_code_transformer:
      # Note that at the moment, we also decode right away inside
      # `range_code`. This means this is slower than it should be.
      output = self._entropy_model_pframe.range_code(
          latent_unquantized=latent,
          previous_latents=state,
          run_decode=frame_index < 5)
    else:
      output = self._entropy_model_pframe(
          latent_unquantized=latent,
          previous_latents=state,
          training=training)

    assert output.features is not None
    bottleneck = Bottleneck(output.perturbed_latent, output.bits,
                            output.features)
    metrics.merge(output.metrics)

    decode_pframe = tf_memoize.bind(self.decode_pframe, cache)
    _, new_state, _ = decode_pframe(
        bottleneck, frame_index, state, training, cache)

    return new_state, EncodeOut(bottleneck, metrics)

  @tf_memoize.memoize
  def decode_pframe(
      self,
      bottleneck,
      frame_index,
      state,
      training,
      cache,
      ):
    latent_q = bottleneck.latent_q
    synthesis_in = self._dequantizer(
        latent_q=latent_q,
        entropy_features=bottleneck.entropy_model_features)
    reconstruction = self._synthesis_image(synthesis_in, training=training)

    # Preprocess `latent_q`.
    next_state_entry = self._entropy_model_pframe.process_previous_latent_q(
        latent_q, training=training)
    new_state = (*state, next_state_entry)
    new_state = new_state[0-self._context_len:]
    return reconstruction, new_state, metric_collection.Metrics.make()

  def encode_frames(
      self,
      frames,
      training,
      cache,
  ):
    state = None
    for frame_index, frame in enumerate(frames):
      if is_iframe(frame_index):
        state, encode_out = self.encode_iframe(frame, training, cache)
      else:
        assert state is not None
        state, encode_out = self.encode_pframe(frame, frame_index, state,
                                               training, cache)
      yield encode_out

  def decode_frames(
      self,
      bottlenecks,
      training,
      cache,
  ):
    decode_iframe = tf_memoize.bind(self.decode_iframe, cache)
    decode_pframe = tf_memoize.bind(self.decode_pframe, cache)

    state = None
    for frame_index, bottleneck in enumerate(bottlenecks):
      if is_iframe(frame_index):
        reconstruction, state, frame_metrics = decode_iframe(
            bottleneck, training)
      else:
        assert state is not None
        reconstruction, state, frame_metrics = decode_pframe(
            bottleneck, frame_index, state, training, cache)
      yield reconstruction, frame_metrics

  def encode_and_decode_frames(
      self,
      frames,
      training,
      cache,
  ):
    """Encodes and decodes frames, and also handles padding/unpadding."""

    (height, width), frames = _spy_spatial_shape(frames)
    frames = _iter_padded(frames, self._pad_factor)

    encode_outs = self.encode_frames(frames, training, cache)

    # Jointly iterate over `encode_outs` twice.
    encode_outs, encode_outs_tee = itertools.tee(encode_outs)
    reconstructions_with_metrics = self.decode_frames(
        (encode_out.bottleneck for encode_out in encode_outs_tee),
        training, cache)

    for ((reconstruction, decode_metrics), encode_out) in zip(
        reconstructions_with_metrics, encode_outs):
      frame_metrics = Metrics.make()
      frame_metrics.merge(encode_out.metrics)
      frame_metrics.merge(decode_metrics)
      reconstruction = reconstruction[:, :height, :width, :]
      yield NetworkOut(
          reconstruction=reconstruction,
          bits=encode_out.bottleneck.bits,
          frame_metrics=frame_metrics,
      )

  def frame_loss(
      self,
      frame,
      network_out,
      training,
  ):
    _, height, width, _ = frame.rgb.shape
    num_pixels = height * width
    bpp_loss = tf.reduce_mean(
        network_out.bits / num_pixels)
    tf.debugging.check_numerics(bpp_loss, "bpp_loss")

    metrics = Metrics.make()

    mse, psnr = _mse_psnr(frame.rgb, network_out.reconstruction, training)
    distortion_loss = mse
    rd_loss = bpp_loss + (self._scheduled_rd_lambda() * distortion_loss)
    metrics.record_image("reconstruction", network_out.reconstruction)
    metrics.record_scalar("mse", mse)
    metrics.record_scalar("psnr", psnr)

    # Check for NaNs in the loss
    tf.debugging.check_numerics(rd_loss, "rd_loss")

    metrics.record_scalar("rd_loss", rd_loss)
    metrics.record_scalar("bpp",
                          tf.reduce_mean(network_out.bits / num_pixels))

    return rd_loss, metrics

  def video_loss(
      self,
      video,
      training,
      cache,
  ):
    """Compute rd loss over a video batch, as well as metrics."""
    video.validate_shape()
    frames = video.get_frames()
    network_outs = self.encode_and_decode_frames(
        frames, training, cache)

    rd_losses = []
    frame_metrics_list = []
    metrics = Metrics.make()
    frame_index = -1  # Prevents `undefined-loop-variable` in assert below.
    for frame_index, (frame, network_out) in enumerate(
        zip(frames, network_outs)):
      rd_loss, rd_metrics = self.frame_loss(
          frame,
          network_out,
          training=training,
      )
      if is_iframe(frame_index):
        rd_loss_weight = 1.0
      else:
        # Use a 10x higher weight for the P-frame.
        rd_loss_weight = 10.0
      rd_loss_scaled = rd_loss * rd_loss_weight
      rd_losses.append(rd_loss_scaled)

      frame_metrics = network_out.frame_metrics
      frame_metrics.merge(rd_metrics)
      frame_metrics.record_scalar("rd_loss_weight", rd_loss_weight)
      frame_metrics.record_scalar("rd_loss_scaled", rd_loss_scaled)
      frame_metrics_list.append(frame_metrics)
      metrics.merge(f"frame_{frame_index}", frame_metrics)

    assert frame_index == video.num_frames - 1, (frame_index,
                                                 video.num_frames - 1)

    video_rd_loss = tf.reduce_mean(rd_losses)
    video_loss = video_rd_loss

    metrics.record_scalar("video_rd_loss", video_rd_loss)
    metrics.record_scalar("video_loss", video_loss)
    metrics.record_scalar("num_frames", video.num_frames)

    metrics.record_scalar("scheduled_rd_lambda",
                          self._scheduled_rd_lambda())
    metrics.record_scalar("scheduled_learning_rate",
                          self._learning_rate_schedule(self.global_step))
    avg_metrics = Metrics.reduce(
        frame_metrics_list, scalar_reduce_fn=tf.reduce_mean)
    metrics.merge("video_avg", avg_metrics)
    return video_loss, metrics

  def _assert_cache_hits(self, num_frames, cache):
    num_iframes = sum(is_iframe(i) for i in range(num_frames))
    num_pframes = num_frames - num_iframes
    def assert_hits(info, expected, actual):
      if expected != actual:
        raise AssertionError(
            f"Expected {expected} hits for {info}, got {actual}!")
    # pytype: disable=attribute-error
    assert_hits("decode_iframe",
                num_iframes, self.decode_iframe.get_total_cache_hits(cache))
    assert_hits("decode_pframe",
                num_pframes, self.decode_pframe.get_total_cache_hits(cache))
    # pytype: enable=attribute-error

  def write_ckpt(self, path, step):
    """Creates a checkpoint at `path` for `step`."""
    ckpt = tf.train.Checkpoint(model=self)
    manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
    manager.save(checkpoint_number=step)
    return tf.train.latest_checkpoint(path)

  def train_step(self, video):
    """Run a training step and return metrics."""
    cache = tf_memoize.create_cache()
    with tf.GradientTape() as tape:
      # This will encode and decode the video, making sure memoize cache is hit.
      video_loss, metrics = self.video_loss(
          video, training=True, cache=cache)
      # The optimizer will sum the gradients across replicas, so we
      # need to scale the loss accordingly.
      local_loss = video_loss / tf.distribute.get_strategy(
      ).num_replicas_in_sync
    var_list = self.all_trainable_variables
    gradients = tape.gradient(local_loss, var_list)
    self._optimizer.apply_gradients(zip(gradients, var_list))
    return metrics

  def _iter_trainable_variables(self):

    def ensure_nonempty(seq):
      if not seq:
        raise ValueError("No trainable variables!")
      return seq

    yield from ensure_nonempty(
        self._temporal_pad_token_maker.trainable_variables)
    yield from ensure_nonempty(self._analysis_image.trainable_variables)
    yield from ensure_nonempty(self._synthesis_image.trainable_variables)
    yield from ensure_nonempty(self._entropy_model_pframe.trainable_variables)
    yield from ensure_nonempty(self._dequantizer.trainable_variables)

  @property
  def all_trainable_variables(self):
    if self._all_trainable_variables is None:
      self._all_trainable_variables = list(
          self._iter_trainable_variables())
      assert self._all_trainable_variables
      assert len(self._all_trainable_variables) == len(self.trainable_variables)
    return self._all_trainable_variables

  def _prepare_for_evaluate(self):
    if not self._did_jit_for_eval:
      if self._range_code_transformer:
        self._entropy_model_pframe.prepare_for_range_coding()

      logging.info("Will jit `entropy_model_pframe`...")
      self._entropy_model_pframe.__call__ = tf.function(jit_compile=True)(
          self._entropy_model_pframe.__call__)
      self._entropy_model_pframe.process_previous_latent_q = tf.function(
          jit_compile=True)(
              self._entropy_model_pframe.process_previous_latent_q)
      self._did_jit_for_eval = True
      logging.info("Did jit `entropy_model_pframe`.")

  def evaluate(self, video):
    """Runs evaluation on a single video.

    NOTE: If `gop_size` is configured for continuous eval or flume eval,
    `video` will actually be a single gop, not the entire video.

    Args:
      video: A video to evaluate

    Yields:
      Eval metrics for each frame.
    """
    self._prepare_for_evaluate()

    frames = video.video
    network_outs = self.encode_and_decode_frames(
        frames, training=False,
        cache=None)  # No memoize cache for eval.

    for i, (frame, network_out) in enumerate(zip(frames, network_outs)):
      logging.info("Compressing frame %d", i)
      _, rd_metrics = self.frame_loss(frame, network_out, training=False)
      network_out.frame_metrics.merge(rd_metrics)
      is_last_frame = (i == len(frames) - 1)
      if is_last_frame:
        network_out.frame_metrics.merge("last_frame", rd_metrics)
      yield network_out.frame_metrics

  def create_variables(self):
    """Creates variables."""
    self._prepare_for_evaluate()
    logging.info("Creating variables...")
    for _ in self.evaluate(
        video_tensors.EvalVideo.make_random(num_frames=3, dim=256)):
      pass
