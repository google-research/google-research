# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A layer which computes Magnitude of RDFT."""
import numpy as np
from kws_streaming.layers import magnitude_rdft
from kws_streaming.layers.compat import tf
import kws_streaming.layers.mel_table as mel_table


class MagnitudeRDFTmel(magnitude_rdft.MagnitudeRDFT):
  """Computes Real DFT Spectrogram and then returns its Magnitude.

  It is useful for speech feature extraction.
  This layer combines Mel spectrum and
  Real Discrete Fourier Transformation(RDFT).
  We merged both pf them, so that the whole transformation
  can be done efficiently by checking non zero values in Mel spectrum
  and reducing RDFT size accordingly.

  Mel spectrum properties are defined by:
  https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix

  We use two implementations of FT one is based on direct DFT,
  which works with TFLite and another is based on TF FFT.

  Attributes:
    use_tf_fft: if True we will use TF FFT otherwise use direct DFT
    which is implemented using matrix matrix multiplications and supported by
    any inference engine.
    magnitude_squared: if True magnitude spectrum will be squared otherwise sqrt
    num_mel_bins: How many bands in the resulting mel spectrum.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
    spectrum. This corresponds to the lower edge of the lowest triangular band.
    upper_edge_hertz: The desired top edge of the highest frequency band.
    sample_rate: Samples per second of the input signal used to
    create the spectrogram.
    mel_non_zero_only: if True we will calculate the non zero area of
    Mel spectrum and use it for DFT calculation to reduce its computations.
  """

  def __init__(self,
               use_tf_fft=False,
               magnitude_squared=False,
               num_mel_bins=40,
               lower_edge_hertz=20.0,
               upper_edge_hertz=4000.0,
               sample_rate=16000.0,
               mel_non_zero_only=True,
               **kwargs):
    super(MagnitudeRDFTmel, self).__init__(
        use_tf_fft=use_tf_fft,
        magnitude_squared=magnitude_squared,
        **kwargs)
    if use_tf_fft and mel_non_zero_only:
      raise ValueError('use_tf_fft and mel_non_zero_only can not be both True')

    self.num_mel_bins = num_mel_bins
    self.lower_edge_hertz = lower_edge_hertz
    self.upper_edge_hertz = upper_edge_hertz
    self.sample_rate = sample_rate
    self.mel_non_zero_only = mel_non_zero_only

  def build(self, input_shape):

    # this is the feature size of the DFT output
    feature_size = int(input_shape[-1])
    if self.use_tf_fft or not self.mel_non_zero_only:
      # this is the feature size of the TF RFFT output
      feature_size = self._compute_fft_size(feature_size) // 2 + 1

    # precompute mel matrix using np
    self.mel_weight_matrix = mel_table.SpectrogramToMelMatrix(
        num_mel_bins=self.num_mel_bins,
        num_spectrogram_bins=feature_size,
        audio_sample_rate=self.sample_rate,
        lower_edge_hertz=self.lower_edge_hertz,
        upper_edge_hertz=self.upper_edge_hertz)

    fft_mel_size = None
    if self.mel_non_zero_only:
      fft_mel_size = self._get_non_zero_mel_size()
      self.mel_weight_matrix = self.mel_weight_matrix[:fft_mel_size, :]

    self.mel_weight_matrix = tf.constant(
        self.mel_weight_matrix, dtype=tf.float32)

    super(MagnitudeRDFTmel, self).build(input_shape, fft_mel_size)

  def call(self, inputs):
    # compute magnitude of fourier spectrum
    fft_mag = super(MagnitudeRDFTmel, self).call(inputs)
    # apply mel spectrum
    return tf.matmul(fft_mag, self.mel_weight_matrix)

  def get_config(self):
    config = {
        'num_mel_bins': self.num_mel_bins,
        'lower_edge_hertz': self.lower_edge_hertz,
        'upper_edge_hertz': self.upper_edge_hertz,
        'sample_rate': self.sample_rate,
        'mel_non_zero_only': self.mel_non_zero_only,
    }
    base_config = super(MagnitudeRDFTmel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_non_zero_mel_size(self):
    """Compute size of mel spectrum with non zero values.

    We can reduce DFT computation by finding the size of area
    of the mel spectrum which is not equal to zero.
    For example if upper_edge_hertz=4000 and sample_rate=16000 we can reduce mel
    matrix by 2x and at the same time reduce DFT size by 2x.
    It will speed up inference proportionally.

    Returns:
      non zero size of mel spectrum
    """
    non_zero_ind = self.mel_weight_matrix.shape[0]
    last_mel_ind = self.mel_weight_matrix.shape[1] - 1
    for i in reversed(range(self.mel_weight_matrix.shape[0])):
      if self.mel_weight_matrix[i, last_mel_ind] != 0.0:
        non_zero_ind = i
        break
    return np.minimum(non_zero_ind + 1, self.mel_weight_matrix.shape[0])
