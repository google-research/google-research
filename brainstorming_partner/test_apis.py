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

"""Checks that the Speech-to-Text API is reachable from a Cloud project.

Sends a short clip of silence to Speech-to-Text V1 and prints whether the call
succeeded. Useful for debugging credentials, IAM permissions, and API
enablement.

Usage:
  python3 test_apis.py --project=YOUR_PROJECT_ID
"""

import asyncio
from absl import app as absl_app
from absl import flags
from google import genai
from google.api_core import client_options as client_options_lib
from google.cloud import speech_v1

_PROJECT = flags.DEFINE_string(
    "project",
    None,
    "Google Cloud project ID to test against.",
    required=True,
)


async def test_stt(project_id):
  """Transcribes 0.1 seconds of silence and prints the outcome."""
  try:
    print("Testing STT API on", project_id)
    # Bill the request to the project under test (needed for user credentials).
    client = speech_v1.SpeechAsyncClient(
        client_options=client_options_lib.ClientOptions(
            quota_project_id=project_id
        )
    )
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    # 3200 zero bytes are 0.1 seconds of 16 kHz, 16-bit mono silence.
    request = speech_v1.RecognizeRequest(
        config=config, audio=speech_v1.RecognitionAudio(content=b"\x00" * 3200)
    )
    await client.recognize(request=request)
    print("STT SUCCESS on", project_id)
  except Exception as e:
    print(
        "STT ERROR on",
        project_id,
        ":",
        type(e).__name__,
        getattr(e, "message", str(e)),
    )


def main(argv):
  asyncio.run(test_stt(_PROJECT.value))


if __name__ == "__main__":
  absl_app.run(main)
