# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Audio processing utilities."""

import io
import os
import subprocess
from typing import Any, Callable, Optional
import wave

from absl import logging
import librosa
import numpy as np


def _decode_wav_with_wave(audio_bytes):
  """Decode WAV bytes using standard wave module."""
  with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
    params = wav_file.getparams()
    n_channels, sampwidth, framerate, n_frames = params[:4]
    raw_data = wav_file.readframes(n_frames)

    if sampwidth == 2:
      data = (
          np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
      )
    elif sampwidth == 1:
      data = (
          np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0
      ) / 128.0
    elif sampwidth == 4:
      data = (
          np.frombuffer(raw_data, dtype=np.int32).astype(np.float32)
          / 2147483648.0
      )
    else:
      raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
      data = data.reshape(-1, n_channels)

    return data, int(framerate)


def _find_ffmpeg():
  """Locate the ffmpeg binary in Bazel runfiles or fall back to system PATH."""
  candidates = [
      "video/vidproc/ffmpeg/sandboxable_ffmpeg",
      "video/common/subprocess/ffmpeg/ffmpeg",
      "third_party/ffmpeg/ffmpeg",
  ]
  # Look under bazel runfiles directories
  for key in ["RUNFILES_DIR", "PYTHON_RUNFILES"]:
    base_dir = os.environ.get(key)
  # Check relative to current working directory
  for c in candidates:
    if os.path.exists(c):
      return c
  # Default to system path
  return "ffmpeg"


def _decode_audio_with_ffmpeg(
    audio_bytes, target_sr = 16000
):
  """Decode arbitrary audio bytes using ffmpeg stdout piping."""
  ffmpeg_cmd = _find_ffmpeg()
  cmd = [
      ffmpeg_cmd,
      "-i",
      "pipe:0",
      "-f",
      "f32le",
      "-acodec",
      "pcm_f32le",
      "-ar",
      str(target_sr),
      "-ac",
      "1",
      "pipe:1",
  ]
  process = subprocess.Popen(
      cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
  )
  out, err = process.communicate(input=audio_bytes)
  if process.returncode != 0:
    raise RuntimeError(f"ffmpeg failed: {err.decode('utf-8')}")
  return np.frombuffer(out, dtype=np.float32)


def _to_mono(audio_data):
  """Converts audio data to mono by averaging channels if necessary."""
  if audio_data.ndim > 1:
    return np.mean(audio_data, axis=1)
  return audio_data


def _resample_if_needed(
    audio_data, source_sr, target_sr
):
  """Resamples audio to target_sr if source_sr is known and different."""
  if target_sr and source_sr and source_sr != target_sr:
    try:
      return librosa.resample(
          audio_data, orig_sr=source_sr, target_sr=target_sr
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("librosa.resample failed: %s", e, exc_info=True)
      # Fallback to returning original audio data if resampling fails
      return audio_data
  return audio_data


def _try_decode_wave(
    audio_bytes, target_sr
):
  """Tries to decode audio using the '_decode_wav_with_wave' function."""
  try:
    audio_data, sr = _decode_wav_with_wave(audio_bytes)
    audio_data = _to_mono(audio_data)
    return _resample_if_needed(audio_data, sr, target_sr)
  except Exception:  # pylint: disable=broad-except
    logging.debug("Decoding with _decode_wav_with_wave failed.", exc_info=True)
    return None




def _try_decode_external_ffmpeg(
    audio_bytes, target_sr
):
  """Tries to decode audio using the external ffmpeg wrapper."""
  try:
    audio_data = _decode_audio_with_ffmpeg(audio_bytes, target_sr or 16000)
    return _to_mono(audio_data)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("External ffmpeg decoder failed: %s", e, exc_info=True)
    return None


def get_audio_array(
    audio_input,
    target_sr = None,
    encoding_extension = "mp3",
):
  """Extracts audio array from standard dict or raw bytes.

  Args:
    audio_input: Audio input, either a dict with 'array'/'bytes' keys, raw
      bytes, or a numpy array.
    target_sr: Optional target sampling rate. If provided and the source
      sampling rate differs, the audio will be resampled.
    encoding_extension: Default encoding extension to assume for raw bytes when
      using the internal decoder (defaults to 'mp3').

  Returns:
    A 1-D float32 numpy array of the audio waveform, or None if decoding fails.
  """
  if isinstance(audio_input, dict):
    if "array" in audio_input:
      array = np.asarray(audio_input["array"], dtype=np.float32)
      source_sr = audio_input.get("sampling_rate")
      array = _to_mono(array)
      return _resample_if_needed(array, source_sr, target_sr)
    if "bytes" in audio_input and audio_input["bytes"] is not None:
      audio_bytes = audio_input["bytes"]
    else:
      raise ValueError("Audio dict contains neither 'array' nor 'bytes'")
  elif isinstance(audio_input, bytes):
    audio_bytes = audio_input
  elif isinstance(audio_input, np.ndarray):
    array = audio_input.astype(np.float32, copy=False)
    array = _to_mono(array)
    if target_sr:
      logging.warning(
          "target_sr specified for np.ndarray input, but source_sr is unknown. "
          "Cannot resample."
      )
    return array
  else:
    raise ValueError(f"Unexpected audio input type: {type(audio_input)}")

  # List of decoder functions to try in order.
  decoder_functions: list[Callable[[], Optional[np.ndarray]]] = [
      lambda: _try_decode_wave(audio_bytes, target_sr),
      lambda: _try_decode_external_ffmpeg(audio_bytes, target_sr),
  ]

  for decoder_func in decoder_functions:
    result = decoder_func()
    if result is not None:
      return result

  logging.error("All audio decoding methods failed for the provided bytes.")
  return None
