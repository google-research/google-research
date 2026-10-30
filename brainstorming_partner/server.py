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

"""FastAPI server for streaming audio via websocket and transcribing it.

The browser UI (index.html) streams microphone audio over a WebSocket. The
server forwards the audio to Cloud Speech-to-Text (STT) for live transcription
and periodically sends the transcript to Gemini (see agent.py), which
summarizes the conversation and decides when to interject with a brainstorming
idea.

Start it with `./run.sh serve --project=YOUR_PROJECT_ID`, which installs
dependencies first, or with `python3 server.py --project=YOUR_PROJECT_ID`.
"""

import asyncio
import inspect
import os
import sys
import time
from typing import Any, Sequence, cast
import uuid

# Disable gRPC's pthread_atfork handlers before importing grpc (via
# google-cloud-speech). When google-auth forks `gcloud` to look up a default
# project while a gRPC polling cycle is active (e.g. an open STT stream on
# macOS), gRPC's fork handler blocks until the RPC deadline and freezes the
# event loop.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# pylint: disable=g-import-not-at-top
# Let `import agent` resolve to the sibling agent.py when run outside of Blaze.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from absl import app as absl_app
from absl import flags
from absl import logging
import agent  # pyrefly: ignore[missing-import]
import fastapi
from fastapi.middleware import cors
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions
from google.cloud import speech_v1
from google.cloud import speech_v2
import uvicorn

# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS

# Command-line flags. Run with --helpshort to list them and their defaults.
flags.DEFINE_integer("port", 8000, "Port to run the FastAPI server on.")
flags.DEFINE_string("host", "0.0.0.0", "Host to run the FastAPI server on.")
flags.DEFINE_string(
    "project",
    None,
    "Required. Google Cloud project ID used for Speech-to-Text and Gemini"
    " (Vertex AI) requests and quota.",
)
flags.DEFINE_string(
    "speech_model",
    "chirp_3",
    "Google Cloud Speech-to-Text model ID (e.g. chirp_3 for V2, or leave"
    " empty/default to use original Speech-to-Text V1 model).",
)
flags.DEFINE_string(
    "speech_location",
    "us",
    "Google Cloud region/location to use for Speech-to-Text V2 API (e.g. us,"
    " us-central1).",
)
flags.DEFINE_string(
    "gemini_model",
    "gemini-3.6-flash",
    "Google Cloud Gemini model ID to use for AI coworker insights.",
)
flags.DEFINE_string(
    "gemini_location",
    "global",
    "Google Cloud region/location to use for Gemini API (e.g. us-central1).",
)
flags.DEFINE_float(
    "insight_interval_sec",
    5.0,
    "Interval in seconds between AI coworker insight requests.",
)
flags.DEFINE_float(
    "interruption_cooldown_sec",
    20.0,
    "Cooldown time in seconds between AI coworker interruptions.",
)
flags.DEFINE_string(
    "chime_sound_path",
    "",
    "Optional path to a notification sound file (.ogg). Defaults to the"
    " bundled notification.ogg.",
)


def _find_resource_path(relative_subpath: str) -> str:
  """Returns the path to a bundled resource file such as index.html.

  Looks next to this module first, then relative to the working directory.
  Falls back to the module-relative path if neither exists.

  Args:
    relative_subpath: File path relative to this module's directory.
  """
  base_dir = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.join(base_dir, relative_subpath),
      os.path.abspath(relative_subpath),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return candidates[0]


def _find_chime_path() -> str:
  """Returns the --chime_sound_path file, or the bundled notification.ogg."""
  chime_flag = FLAGS["chime_sound_path"].value
  if chime_flag:
    return os.path.abspath(chime_flag)
  return _find_resource_path("notification.ogg")


def is_idle_or_stream_timeout(exc: Exception) -> bool:
  """Returns True if the exception represents an idle/silence timeout or stream expiration."""
  if isinstance(
      exc,
      (
          exceptions.DeadlineExceeded,
          exceptions.OutOfRange,
          exceptions.Aborted,
          exceptions.ServiceUnavailable,
      ),
  ):
    return True
  # Some timeouts arrive as generic errors, so also match known message text.
  err_str = str(exc).lower()
  idle_keywords = (
      "stream timed out after receiving no more client requests",
      "stream timed out",
      "no more client requests",
      "exceeded maximum allowed stream duration",
      "maximum allowed stream duration",
      "stream duration limit exceeded",
      "out of range: audio data duration",
      "out of range",
  )
  return any(kw in err_str for kw in idle_keywords)


def format_timestamp(seconds: float | int) -> str:
  """Formats seconds into a [MM:SS] timestamp string."""
  total_sec = max(0, int(seconds))
  return f"[{total_sec // 60:02d}:{total_sec % 60:02d}]"


# Single-page web UI served for every session.
INDEX_HTML_PATH = _find_resource_path("index.html")


class SessionState:
  """Holds the state for a single active or resumed session.

  Sessions are kept in memory (see SESSIONS), so a browser can reconnect or
  reload the page without losing the conversation.

  Attributes:
    session_id: Unique ID that appears in the session URL.
    timestamped_transcript_lines: Finalized transcript lines prefixed with
      [MM:SS] timestamps, plus AI suggestions and pause/resume/reload markers.
    latest_transcript: Joined transcript, including the newest interim result.
    history: Completed segments (summary and optional AI suggestion) and system
      events such as pause, resume, and reload.
    latest_summary: Gemini's summary of the in-progress segment.
    latest_summary_range: "[MM:SS - MM:SS]" range covered by latest_summary.
    latest_summary_timestamp: When latest_summary was generated.
    segment_start_timestamp: Start of the in-progress segment, which begins at
      the last AI suggestion, pause, or resume.
    last_interruption_line_index: Index into timestamped_transcript_lines just
      after the last AI suggestion. Gemini analyzes the lines from here on.
    last_processed_transcript: Transcript last sent to Gemini, used to skip
      duplicate requests.
    is_paused: Whether the user paused listening.
    stream_start_time: time.perf_counter() value when the session clock started;
      timestamps are relative to it.
    total_bytes: Total audio bytes received from the browser.
    active_websocket: The browser connection currently attached, if any.
    stt_task: Background task that streams audio to Speech-to-Text.
    gemini_task: Background task that periodically requests Gemini insights.
  """

  def __init__(self, session_id: str):
    self.session_id = session_id
    self.timestamped_transcript_lines: list[str] = []
    self.latest_transcript: str = ""
    self.history: list[dict[str, Any]] = []
    self.latest_summary: str = ""
    self.latest_summary_range: str = ""
    self.latest_summary_timestamp: str = ""
    self.segment_start_timestamp: str = "[00:00]"
    self.last_interruption_line_index: int = 0
    self.last_processed_transcript: str = ""
    self.is_paused: bool = False
    self.stream_start_time: float | None = None
    self.total_bytes: int = 0
    self.active_websocket: fastapi.WebSocket | None = None
    self.stt_task: asyncio.Task[Any] | None = None
    self.gemini_task: asyncio.Task[Any] | None = None


def ensure_session_start_time(session: SessionState) -> float:
  """Starts the session clock on first use and returns its start time."""
  if session.stream_start_time is None:
    session.stream_start_time = time.perf_counter()
    session.segment_start_timestamp = "[00:00]"
  return session.stream_start_time


# All sessions by ID. Kept in memory only, so they are lost on restart.
SESSIONS: dict[str, SessionState] = {}


def find_index_html() -> str:
  """Returns the index.html path, looking it up again if it has moved."""
  if os.path.exists(INDEX_HTML_PATH):
    return INDEX_HTML_PATH
  return _find_resource_path("index.html")


app = fastapi.FastAPI(
    title="Audio Stream Transcriber & AI Brainstorming Coworker"
)

# Allow requests from any origin. The server has no authentication, so only run
# it on trusted networks.
app.add_middleware(
    cors.CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=fastapi.responses.RedirectResponse)
async def get_index():
  """Creates a new session and redirects the browser to its page."""
  session_id = str(uuid.uuid4())
  SESSIONS[session_id] = SessionState(session_id)
  return fastapi.responses.RedirectResponse(
      url=f"/session/{session_id}",
      status_code=fastapi.status.HTTP_303_SEE_OTHER,
  )


@app.get("/session/{session_id}")
async def get_session_page(session_id: str):
  """Serves the UI for a session; unknown IDs redirect to a new session."""
  if session_id not in SESSIONS:
    return fastapi.responses.RedirectResponse(
        url="/", status_code=fastapi.status.HTTP_303_SEE_OTHER
    )
  index_path = find_index_html()
  if os.path.exists(index_path):
    return fastapi.responses.FileResponse(index_path, media_type="text/html")
  raise fastapi.HTTPException(status_code=404, detail="index.html not found")


@app.get("/audio/notification.ogg")
async def get_chime_sound():
  """Serves the chime the UI plays when the AI coworker has a suggestion."""
  chime_path = _find_chime_path()
  if os.path.exists(chime_path):
    return fastapi.responses.FileResponse(chime_path, media_type="audio/ogg")
  raise fastapi.HTTPException(status_code=404, detail="Sound file not found")


def create_stt_client(
    project_id: str,
    speech_model: str | None = None,
    speech_location: str | None = None,
) -> tuple[speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient, bool]:
  """Creates a SpeechAsyncClient (V2 or V1) based on speech_model."""
  if speech_model is None:
    speech_model = FLAGS.speech_model
  if speech_location is None:
    speech_location = FLAGS.speech_location

  # Chirp models are only available through the V2 API, which is served from
  # regional endpoints (e.g. us-speech.googleapis.com). Both API versions bill
  # usage to project_id as the quota project.
  use_v2 = bool(speech_model and speech_model.lower().startswith("chirp"))
  if use_v2:
    loc = speech_location if speech_location else "us"
    api_endpoint = f"{loc}-speech.googleapis.com" if loc != "global" else None
    client = speech_v2.SpeechAsyncClient(
        client_options=client_options_lib.ClientOptions(
            api_endpoint=api_endpoint,
            quota_project_id=project_id,
        )
    )
  else:
    client = speech_v1.SpeechAsyncClient(
        client_options=client_options_lib.ClientOptions(
            quota_project_id=project_id
        )
    )
  return client, use_v2


async def _close_stt_client(
    client: speech_v1.SpeechAsyncClient | speech_v2.SpeechAsyncClient | None,
) -> None:
  """Closes a SpeechAsyncClient's underlying gRPC transport, if any."""
  if client is None:
    return
  transport = getattr(client, "transport", None)
  close_fn = getattr(transport, "close", None)
  if callable(close_fn):
    try:
      close_res = close_fn()
      if inspect.isawaitable(close_res):
        await close_res
    except Exception:  # pylint: disable=broad-except
      pass


def make_stt_audio_request(
    chunk: bytes, use_v2: bool
) -> speech_v1.StreamingRecognizeRequest | speech_v2.StreamingRecognizeRequest:
  """Creates a StreamingRecognizeRequest for an audio chunk (V2 or V1)."""
  if use_v2:
    return speech_v2.StreamingRecognizeRequest(audio=chunk)
  return speech_v1.StreamingRecognizeRequest(audio_content=chunk)


async def check_stt_health(
    project_id: str,
    speech_model: str | None = None,
    speech_location: str | None = None,
) -> dict[str, str]:
  """Checks Speech-to-Text access by transcribing 0.1 seconds of silence.

  Args:
    project_id: Google Cloud project to call and bill.
    speech_model: Speech model ID. Defaults to --speech_model.
    speech_location: Speech-to-Text V2 location. Defaults to --speech_location.

  Returns:
    A dict with "status" ("live" or "error"), "model", and, on failure,
    "error".
  """
  if speech_model is None:
    speech_model = FLAGS.speech_model
  if speech_location is None:
    speech_location = FLAGS.speech_location

  model_display = speech_model if speech_model else "default (V1)"

  client = None
  try:
    client, use_v2 = create_stt_client(
        project_id=project_id,
        speech_model=speech_model,
        speech_location=speech_location,
    )
    if use_v2:
      client_v2 = cast(speech_v2.SpeechAsyncClient, client)
      loc = speech_location if speech_location else "us"
      config = speech_v2.RecognitionConfig(
          explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
              encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
              sample_rate_hertz=16000,
              audio_channel_count=1,
          ),
          language_codes=["en-US"],
          model=speech_model,
      )
      request = speech_v2.RecognizeRequest(
          recognizer=f"projects/{project_id}/locations/{loc}/recognizers/_",
          config=config,
          content=b"\x00" * 3200,
      )
      await client_v2.recognize(request=request)
    else:
      client_v1 = cast(speech_v1.SpeechAsyncClient, client)
      v1_model = (
          speech_model
          if speech_model
          and speech_model.lower() not in ("default", "original", "v1")
          else None
      )
      config = speech_v1.RecognitionConfig(
          encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
          sample_rate_hertz=16000,
          language_code="en-US",
          model=v1_model,
      )
      audio = speech_v1.RecognitionAudio(content=b"\x00" * 3200)
      await client_v1.recognize(config=config, audio=audio)
    return {"status": "live", "model": f"Speech-to-Text {model_display}"}
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("check_stt_health failed: %s", e)
    return {
        "status": "error",
        "model": f"Speech-to-Text {model_display}",
        "error": str(e),
    }
  finally:
    await _close_stt_client(client)


async def check_gemini_health(
    brainstorm_agent: agent.BrainstormAgent,
) -> dict[str, str]:
  """Checks Gemini access by sending a short prompt.

  Args:
    brainstorm_agent: Agent whose Gemini client and model are checked.

  Returns:
    A dict in the same format as check_stt_health().
  """
  try:
    await brainstorm_agent.health_check()
    return {"status": "live", "model": brainstorm_agent.model_name}
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("check_gemini_health failed: %s", e)
    return {
        "status": "error",
        "model": brainstorm_agent.model_name,
        "error": str(e),
    }


# Cache healthy /api/status results for 60s and share in-flight checks so
# concurrent or frequent UI status polls do not repeatedly create clients,
# spawn `gcloud` via google-auth, and hit Speech-to-Text and Gemini.
_API_STATUS_CACHE_TTL_SEC = 60.0
_api_status_cache: (
    tuple[tuple[Any, ...], float, dict[str, str], dict[str, str]] | None
) = None
_api_status_in_flight: dict[
    tuple[Any, ...], asyncio.Task[tuple[dict[str, str], dict[str, str]]]
] = {}


def clear_api_status_cache() -> None:
  """Clears the cached /api/status health check results."""
  global _api_status_cache
  _api_status_cache = None
  _api_status_in_flight.clear()


@app.get("/api/status")
async def get_api_status(session_id: str | None = None):
  """Reports API health and, if session_id is given, its task state.

  The UI polls this to show the Speech-to-Text and Gemini status badges and to
  restart a session whose background tasks stopped.
  """
  global _api_status_cache
  cache_key = (
      FLAGS.project,
      FLAGS.speech_model,
      FLAGS.speech_location,
      FLAGS.gemini_model,
      FLAGS.gemini_location,
  )
  now = time.monotonic()
  if (
      _api_status_cache is not None
      and _api_status_cache[0] == cache_key
      and now - _api_status_cache[1] < _API_STATUS_CACHE_TTL_SEC
  ):
    stt_res, gemini_res = _api_status_cache[2], _api_status_cache[3]
  else:
    in_flight = _api_status_in_flight.get(cache_key)
    if in_flight is None or in_flight.done():

      async def _run_checks() -> tuple[dict[str, str], dict[str, str]]:
        brainstorm_agent = agent.BrainstormAgent(
            project=FLAGS.project,
            model_name=FLAGS.gemini_model,
            location=FLAGS.gemini_location,
        )
        return await asyncio.gather(
            check_stt_health(
                FLAGS.project,
                speech_model=FLAGS.speech_model,
                speech_location=FLAGS.speech_location,
            ),
            check_gemini_health(brainstorm_agent),
        )

      in_flight = asyncio.create_task(_run_checks())
      _api_status_in_flight[cache_key] = in_flight
    try:
      stt_res, gemini_res = await in_flight
    finally:
      if _api_status_in_flight.get(cache_key) is in_flight:
        _api_status_in_flight.pop(cache_key, None)
    if stt_res.get("status") == "live" and gemini_res.get("status") == "live":
      _api_status_cache = (
          cache_key,
          time.monotonic(),
          stt_res,
          gemini_res,
      )

  session_info = None
  if session_id and session_id in SESSIONS:
    s = SESSIONS[session_id]
    session_info = {
        "is_paused": s.is_paused,
        "stt_task_running": bool(
            s.stt_task is not None and not s.stt_task.done()
        ),
        "gemini_task_running": bool(
            s.gemini_task is not None and not s.gemini_task.done()
        ),
    }

  return {
      "stt": stt_res,
      "gemini": gemini_res,
      "session": session_info,
  }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: fastapi.WebSocket, session_id: str):
  """Streams a session's audio to Speech-to-Text and Gemini.

  Browser-to-server messages:
    * Binary: WebM/Opus audio chunks recorded from the microphone.
    * Text: "PING" (heartbeat), "PAUSE", "RESUME", or "STOP".

  Server-to-browser messages are JSON objects with transcript updates, Gemini
  summaries and suggestions, latency metrics, and errors.

  Args:
    websocket: The browser connection.
    session_id: Session to attach to; created if it does not exist.
  """
  session = SESSIONS.get(session_id)
  if session is None:
    session = SessionState(session_id)
    SESSIONS[session_id] = session

  # If this websocket takes over another one for the same session, close the other one.
  if (
      session.active_websocket is not None
      and session.active_websocket != websocket
  ):
    logging.info(
        "Closing existing active WebSocket connection for session %s",
        session_id,
    )
    try:
      await session.active_websocket.close(
          code=1000, reason="Resumed in another connection"
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Error closing old websocket: %s", e)

  session.active_websocket = websocket
  await websocket.accept()

  # Mark reconnects (e.g. a page reload) in the transcript and history.
  is_reload = bool(session.latest_transcript or session.history)
  if is_reload:
    stream_start_time = ensure_session_start_time(session)
    now_sec = time.perf_counter() - stream_start_time
    ts = format_timestamp(now_sec)
    session.history.append({
        "system_event": "reload",
        "timestamp": ts,
        "text": f"{ts} Session Reloaded",
    })
    if not session.timestamped_transcript_lines and session.latest_transcript:
      session.timestamped_transcript_lines = session.latest_transcript.split(
          "\n"
      )
    session.timestamped_transcript_lines.append(f"{ts} Session Reloaded")
    session.latest_transcript = "\n".join(session.timestamped_transcript_lines)

  # If there is existing state, send it immediately to synchronize the client
  if session.latest_transcript or session.history:
    try:
      await websocket.send_json({
          "transcript": session.latest_transcript,
          "history": list(session.history),
          "current_summary": session.latest_summary,
          "current_summary_range": session.latest_summary_range,
      })
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Failed to send initial session state: %s", e)

  # (chunk, receive_time) tuples consumed by the STT stream; None means stop.
  audio_queue = asyncio.Queue()
  # Serializes sends; the STT and Gemini tasks write to the socket concurrently.
  ws_lock = asyncio.Lock()
  # Time and size of the last chunk sent to STT, for latency metrics.
  last_chunk_yield_time = None
  last_chunk_bytes = 0
  logging.info(
      "WebSocket connection accepted from %s for session %s",
      websocket.client,
      session_id,
  )

  # The first audio chunk holds the WebM header. It is replayed at the start of
  # each new STT stream so later chunks can still be decoded.
  header_chunk: bytes | None = None
  # Number of STT streams opened by this connection.
  stream_counter = 0

  # When the AI coworker last interjected, for --interruption_cooldown_sec.
  last_interruption_time = None
  # History length at the last Gemini update; a change triggers a new one.
  last_processed_history_count = len(session.history)

  async def safe_send_json(data: dict[str, Any]):
    """Sends JSON to the browser, logging instead of raising on failure."""
    async with ws_lock:
      try:
        await websocket.send_json(data)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to send JSON over WebSocket: %s", e)

  async def request_generator():
    """Yields the requests for one STT streaming call.

    Sends the recognition config, replays the WebM header on reconnects, then
    forwards audio chunks until the stream nears the API's duration limit.
    """
    nonlocal last_chunk_yield_time, last_chunk_bytes, stream_counter
    stream_counter += 1
    current_stream_id = stream_counter

    # Wait for the first audio chunk from audio_queue before yielding any requests.
    # This prevents STT from timing out when the session is idle/silent before audio begins.
    first_item = await audio_queue.get()
    if first_item is None:
      # Audio queue was closed or task stopping
      return

    first_chunk, t_recv = first_item
    stream_start_time_local = time.perf_counter()

    use_v2 = bool(
        FLAGS.speech_model and FLAGS.speech_model.lower().startswith("chirp")
    )

    if use_v2:
      # V2: the first request names the recognizer and carries the config.
      loc = FLAGS.speech_location if FLAGS.speech_location else "us"
      config = speech_v2.RecognitionConfig(
          explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
              encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.WEBM_OPUS,
              sample_rate_hertz=48000,
              audio_channel_count=1,
          ),
          language_codes=["en-US"],
          model=FLAGS.speech_model,
          features=None,
      )
      streaming_config = speech_v2.StreamingRecognitionConfig(
          config=config,
          streaming_features=speech_v2.StreamingRecognitionFeatures(
              interim_results=True,
          ),
      )
      logging.info(
          "request_generator [%d]: Yielding V2 StreamingRecognitionConfig"
          " (model=%s, location=%s)",
          current_stream_id,
          FLAGS.speech_model,
          loc,
      )
      yield speech_v2.StreamingRecognizeRequest(
          recognizer=f"projects/{FLAGS.project}/locations/{loc}/recognizers/_",
          streaming_config=streaming_config,
      )

      if stream_counter > 1 and header_chunk is not None:
        yield make_stt_audio_request(header_chunk, use_v2)

      last_chunk_yield_time = time.perf_counter()
      last_chunk_bytes = len(first_chunk)
      yield make_stt_audio_request(first_chunk, use_v2)
    else:
      # V1: the first request carries only the streaming config.
      v1_model = (
          FLAGS.speech_model
          if FLAGS.speech_model
          and FLAGS.speech_model.lower() not in ("default", "original", "v1")
          else None
      )
      config = speech_v1.RecognitionConfig(
          encoding=speech_v1.RecognitionConfig.AudioEncoding.WEBM_OPUS,
          sample_rate_hertz=48000,
          language_code="en-US",
          model=v1_model,
          diarization_config=None,
      )
      streaming_config = speech_v1.StreamingRecognitionConfig(
          config=config,
          interim_results=True,
      )
      logging.info(
          "request_generator [%d]: Yielding V1 StreamingRecognitionConfig"
          " (model=%s)",
          current_stream_id,
          v1_model,
      )
      yield speech_v1.StreamingRecognizeRequest(
          streaming_config=streaming_config
      )

      if stream_counter > 1 and header_chunk is not None:
        yield make_stt_audio_request(header_chunk, use_v2)

      last_chunk_yield_time = time.perf_counter()
      last_chunk_bytes = len(first_chunk)
      yield make_stt_audio_request(first_chunk, use_v2)

    chunk_idx = 1
    while True:
      # Speech-to-Text caps streams at ~5 minutes, so end this one after 4;
      # process_stt() then opens a new stream.
      if time.perf_counter() - stream_start_time_local > 240.0:
        logging.info(
            "request_generator [%d]: Stream duration reached threshold;"
            " ending stream.",
            current_stream_id,
        )
        break

      # Wake at least once a second so the duration check runs during silence.
      try:
        item = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
      except asyncio.TimeoutError:
        continue

      if item is None:
        break
      chunk, t_recv = item
      t_yield = time.perf_counter()
      last_chunk_yield_time = t_yield
      last_chunk_bytes = len(chunk)

      chunk_idx += 1
      if chunk_idx <= 5 or chunk_idx % 20 == 0:
        logging.info(
            "request_generator [%d]: Yielding audio chunk #%d (%d bytes)",
            current_stream_id,
            chunk_idx,
            len(chunk),
        )
      yield make_stt_audio_request(chunk, use_v2)

  async def process_stt():
    """Runs STT streams and forwards transcripts to the browser.

    Opens a new stream whenever one ends or times out from silence, and retries
    other errors with exponential backoff, giving up after max_retries.
    """
    logging.info(
        "process_stt: Initializing Speech client for project '%s'...",
        FLAGS.project,
    )
    retry_count = 0
    max_retries = 10
    base_delay = 1.0

    while retry_count < max_retries:
      client = None
      try:
        client, use_v2 = create_stt_client(
            project_id=FLAGS.project,
            speech_model=FLAGS.speech_model,
            speech_location=FLAGS.speech_location,
        )
        logging.info(
            "process_stt: Calling client.streaming_recognize (attempt"
            " %d/%d)...",
            retry_count + 1,
            max_retries,
        )
        responses = await client.streaming_recognize(
            requests=request_generator()
        )
        logging.info("process_stt: Stream established. Awaiting responses...")
        async for response in responses:
          retry_count = 0  # Reset retry count on successful response
          t_resp_recv = time.perf_counter()
          stt_latency_ms = (
              (t_resp_recv - last_chunk_yield_time) * 1000
              if last_chunk_yield_time
              else 0.0
          )
          stream_start_time = ensure_session_start_time(session)
          total_elapsed_ms = (t_resp_recv - stream_start_time) * 1000

          logging.info(
              "process_stt: STT Response received | STT API Latency: %.2f ms |"
              " Total Stream Time: %.2f ms | Results: %d",
              stt_latency_ms,
              total_elapsed_ms,
              len(response.results),
          )
          for result in response.results:
            if not result.alternatives:
              logging.info("process_stt: Result has no alternatives.")
              continue
            alt = result.alternatives[0].transcript
            logging.info(
                "process_stt: Result transcript: '%s' | is_final=%s |"
                " stability=%.2f | STT API Latency: %.2f ms",
                alt,
                result.is_final,
                result.stability,
                stt_latency_ms,
            )

            elapsed_sec = t_resp_recv - stream_start_time
            timestamp_prefix = format_timestamp(elapsed_sec)

            # Final results are appended to the transcript. Interim results are
            # shown with a "(transcribing...)" suffix until they are finalized.
            if result.is_final:
              session.timestamped_transcript_lines.append(
                  f"{timestamp_prefix} {alt}"
              )
              session.latest_transcript = "\n".join(
                  session.timestamped_transcript_lines
              )
              display_transcript = session.latest_transcript
            else:
              interim_line = f"{timestamp_prefix} {alt} (transcribing...)"
              display_transcript = "\n".join(
                  session.timestamped_transcript_lines + [interim_line]
              )
              session.latest_transcript = "\n".join(
                  session.timestamped_transcript_lines
                  + [f"{timestamp_prefix} {alt}"]
              )

            if display_transcript:
              await safe_send_json({
                  "transcript": display_transcript,
                  "is_final": result.is_final,
                  "stt_latency_ms": round(stt_latency_ms, 1),
                  "last_chunk_bytes": last_chunk_bytes,
              })
        logging.info(
            "process_stt: Response stream finished normally; rotating stream..."
        )
        retry_count = 0
        continue
      except asyncio.CancelledError:
        logging.info("process_stt: Task cancelled.")
        break
      except Exception as e:  # pylint: disable=broad-except
        # Silence and stream-duration timeouts are expected; reconnect quietly.
        # TODO(beekley): Remove ServiceUnavailable from
        # is_idle_or_stream_timeout so outages and bad --speech_location values
        # back off and hit max_retries instead of retrying every 50 ms forever.
        if is_idle_or_stream_timeout(e):
          logging.info(
              "process_stt: STT stream closed due to inactivity or stream"
              " timeout (%s). Ready to re-establish on next audio chunk.",
              e,
          )
          retry_count = 0
          await asyncio.sleep(0.05)
          continue

        retry_count += 1
        logging.exception(
            "process_stt: Exception in STT stream (attempt %d/%d): %s",
            retry_count,
            max_retries,
            e,
        )
        if retry_count >= max_retries:
          suggestion = (
              "Make sure Application Default Credentials are configured by"
              " running `gcloud auth application-default login` on your"
              " system. Also verify that the Cloud Speech-to-Text API is"
              f" enabled in your Google Cloud Project '{FLAGS.project}'."
          )
          await safe_send_json({"error": str(e), "suggestion": suggestion})
          break
        await safe_send_json({
            "stt_warning": (
                f"STT API error ({e}). Retrying stream (attempt"
                f" {retry_count}/{max_retries})..."
            )
        })
        # Tests use a tiny --insight_interval_sec; back off quickly there too.
        base_delay_val = (
            0.01 if FLAGS.insight_interval_sec < 0.1 else base_delay
        )
        await asyncio.sleep(
            min(base_delay_val * (2 ** (retry_count - 1)), 10.0)
        )
      finally:
        await _close_stt_client(client)

  async def process_gemini_insights():
    """Periodically asks Gemini for a conversation summary and suggestions.

    Every --insight_interval_sec, sends the transcript since the last AI
    suggestion (plus earlier segment history) to Gemini. When Gemini chooses to
    interject and the cooldown has passed, records the suggestion, closes the
    current segment, and starts a new one.
    """
    nonlocal last_interruption_time, last_processed_history_count
    logging.info(
        "process_gemini_insights: Initializing BrainstormAgent (project='%s', "
        "model='%s', location='%s')...",
        FLAGS.project,
        FLAGS.gemini_model,
        FLAGS.gemini_location,
    )
    try:
      brainstorm_agent = agent.BrainstormAgent(
          project=FLAGS.project,
          model_name=FLAGS.gemini_model,
          location=FLAGS.gemini_location,
      )
      while True:
        await asyncio.sleep(FLAGS.insight_interval_sec)
        if session.is_paused:
          continue
        current_transcript_lines = session.timestamped_transcript_lines[
            session.last_interruption_line_index :
        ]
        transcript_to_summarize = "\n".join(current_transcript_lines).strip()

        # default values to appease pytype
        agent_response = None
        gemini_latency_ms = 0.0
        summary = ""
        should_interrupt = False
        interrupt_text = ""

        if not transcript_to_summarize:
          effective_transcript = ""
          # Skip hitting the real Gemini API if there is no pending speech
          display_summary = (
              "Listening carefully, a conversation summary will appear soon."
              if not session.history
              else "(No summary formulated yet - session active)"
          )
          agent_response = agent.BrainstormResponse(
              summary=display_summary, should_interrupt=False, interrupt_text=""
          )
          gemini_latency_ms = 0.0
        else:
          effective_transcript = transcript_to_summarize

        # Only send an update if the transcript or history changed since the
        # last one.
        if (
            effective_transcript != session.last_processed_transcript
            or len(session.history) != last_processed_history_count
        ):
          try:
            if effective_transcript:
              logging.info(
                  "process_gemini_insights: Analyzing transcript snippet (%d"
                  " chars, history_count=%d) with Gemini model '%s'...",
                  len(effective_transcript),
                  len(session.history),
                  FLAGS.gemini_model,
              )
              await safe_send_json({
                  "gemini_request": {
                      "model": FLAGS.gemini_model,
                      "char_count": len(effective_transcript),
                      "transcript_snippet": effective_transcript,
                      "history_count": len(session.history),
                  }
              })
              agent_response, gemini_latency_ms = (
                  await brainstorm_agent.analyze_transcript(
                      effective_transcript, history=list(session.history)
                  )
              )
            if agent_response:
              summary = agent_response.summary
              should_interrupt = agent_response.should_interrupt
              interrupt_text = agent_response.interrupt_text

            logging.info(
                "process_gemini_insights: Gemini Response received (latency:"
                " %.2f ms) | summary: '%s' | should_interrupt: %s |"
                " interrupt_text: '%s'",
                gemini_latency_ms,
                summary,
                should_interrupt,
                interrupt_text,
            )
            session.last_processed_transcript = effective_transcript
            last_processed_history_count = len(session.history)

            now_sec = time.perf_counter() - ensure_session_start_time(session)
            current_timestamp = format_timestamp(now_sec)
            start_t = session.segment_start_timestamp.strip("[]")
            end_t = current_timestamp.strip("[]")
            summary_range = f"[{start_t} - {end_t}]"

            session.latest_summary = summary
            session.latest_summary_range = summary_range
            session.latest_summary_timestamp = current_timestamp

            if should_interrupt and interrupt_text:
              current_time = time.perf_counter()
              # Check if cooldown is over.
              if (
                  last_interruption_time is not None
                  and (current_time - last_interruption_time)
                  < FLAGS.interruption_cooldown_sec
              ):
                logging.info(
                    "process_gemini_insights: Interruption suppressed due to"
                    " cooldown (%.1fs < %.1fs cooldown).",
                    current_time - last_interruption_time,
                    FLAGS.interruption_cooldown_sec,
                )
                should_interrupt = False
                interrupt_text = ""
              else:
                # Record the suggestion as a completed segment and start a new
                # segment from here.
                last_interruption_time = current_time
                timestamp_prefix = current_timestamp
                session.history.append({
                    "summary": summary,
                    "suggestion": interrupt_text,
                    "timestamp": timestamp_prefix,
                    "start_timestamp": session.segment_start_timestamp,
                    "end_timestamp": current_timestamp,
                    "timestamp_range": summary_range,
                })
                agent_line = f"{timestamp_prefix} AI Coworker: {interrupt_text}"
                session.timestamped_transcript_lines.append(agent_line)
                session.latest_transcript = "\n".join(
                    session.timestamped_transcript_lines
                )
                session.last_interruption_line_index = len(
                    session.timestamped_transcript_lines
                )
                session.segment_start_timestamp = current_timestamp
                session.last_processed_transcript = ""
                session.latest_summary = ""
                last_processed_history_count = len(session.history)

            await safe_send_json({
                "transcript": session.latest_transcript,
                "history": list(session.history),
                "current_summary": summary if not should_interrupt else "",
                "current_summary_range": (
                    summary_range if not should_interrupt else ""
                ),
                "gemini_summary": summary,
                "should_interrupt": should_interrupt,
                "interrupt_text": interrupt_text,
                "gemini_latency_ms": round(gemini_latency_ms, 1),
            })
          except Exception as e:  # pylint: disable=broad-except
            logging.exception(
                "process_gemini_insights: Error generating Gemini insights for"
                " model '%s': %s",
                FLAGS.gemini_model,
                e,
            )
            await safe_send_json({"gemini_error": str(e)})
    except asyncio.CancelledError:
      logging.info("process_gemini_insights: Task cancelled.")
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          "process_gemini_insights: Unexpected initialization error in"
          " process_gemini_insights: %s",
          e,
      )
      await safe_send_json({"gemini_error": str(e)})

  async def record_system_event(event_type: str, display_text: str):
    """Records a pause or resume event and sends the update to the browser.

    Args:
      event_type: "pause" or "resume".
      display_text: Text shown in the transcript, e.g. "Session Paused".
    """
    nonlocal last_processed_history_count
    stream_start_time = ensure_session_start_time(session)
    session.is_paused = event_type == "pause"
    now_sec = time.perf_counter() - stream_start_time
    ts = format_timestamp(now_sec)

    # Save the in-progress summary as its own segment so pausing keeps it.
    if event_type == "pause" and session.latest_summary:
      start_t = session.segment_start_timestamp.strip("[]")
      end_t = ts.strip("[]")
      session.history.append({
          "summary": session.latest_summary,
          "timestamp": ts,
          "start_timestamp": session.segment_start_timestamp,
          "end_timestamp": ts,
          "timestamp_range": f"[{start_t} - {end_t}]",
      })
      session.latest_summary = ""

    session.history.append({
        "system_event": event_type,
        "timestamp": ts,
        "text": f"{ts} {display_text}",
    })
    if not session.timestamped_transcript_lines and session.latest_transcript:
      session.timestamped_transcript_lines = session.latest_transcript.split(
          "\n"
      )
    session.timestamped_transcript_lines.append(f"{ts} {display_text}")
    session.latest_transcript = "\n".join(session.timestamped_transcript_lines)
    session.segment_start_timestamp = ts
    if event_type == "resume":
      session.last_processed_transcript = ""

    last_processed_history_count = len(session.history)

    await safe_send_json({
        "system_event": event_type,
        "timestamp": ts,
        "history": list(session.history),
        "current_summary": "",
        "transcript": session.latest_transcript,
    })

  def start_background_tasks():
    """Starts fresh STT and Gemini tasks, cancelling any that are running."""
    if session.stt_task is not None and not session.stt_task.done():
      session.stt_task.cancel()
      session.stt_task = None
    if session.gemini_task is not None and not session.gemini_task.done():
      session.gemini_task.cancel()
      session.gemini_task = None
    # Clear the queue first
    while not audio_queue.empty():
      try:
        audio_queue.get_nowait()
      except asyncio.QueueEmpty:
        break
    logging.info(
        "Starting background STT and Gemini tasks for session %s", session_id
    )
    session.stt_task = asyncio.create_task(process_stt())
    session.gemini_task = asyncio.create_task(process_gemini_insights())

  async def stop_background_tasks():
    """Cancels the STT and Gemini tasks and waits for them to exit."""
    tasks = []
    if session.stt_task is not None:
      session.stt_task.cancel()
      tasks.append(session.stt_task)
      session.stt_task = None
    if session.gemini_task is not None:
      session.gemini_task.cancel()
      tasks.append(session.gemini_task)
      session.gemini_task = None
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)

  def ensure_background_tasks_running():
    """Restarts background tasks that exited while the session is active."""
    if not session.is_paused:
      if session.stt_task is None or session.stt_task.done():
        if (
            session.stt_task is not None
            and session.stt_task.done()
            and not session.stt_task.cancelled()
            and session.stt_task.exception()
        ):
          logging.warning(
              "STT task for session %s failed with exception: %s."
              " Restarting...",
              session_id,
              session.stt_task.exception(),
          )
        else:
          logging.info(
              "STT task for active session %s was done; restarting...",
              session_id,
          )
        session.stt_task = asyncio.create_task(process_stt())
      if session.gemini_task is None or session.gemini_task.done():
        if (
            session.gemini_task is not None
            and session.gemini_task.done()
            and not session.gemini_task.cancelled()
            and session.gemini_task.exception()
        ):
          logging.warning(
              "Gemini task for session %s failed with exception: %s."
              " Restarting...",
              session_id,
              session.gemini_task.exception(),
          )
        else:
          logging.info(
              "Gemini task for active session %s was done; restarting...",
              session_id,
          )
        session.gemini_task = asyncio.create_task(process_gemini_insights())

  if not session.is_paused:
    start_background_tasks()

  # Route audio and control messages from the browser until it disconnects.
  try:
    while True:
      message = await websocket.receive()
      if message.get("type") == "websocket.disconnect":
        logging.info("WebSocket client disconnected.")
        break
      ensure_background_tasks_running()
      if "bytes" in message:
        ensure_session_start_time(session)
        if session.is_paused:
          continue
        chunk = message["bytes"]
        if chunk:
          # Keep the first chunk (the WebM header) for STT reconnects.
          if header_chunk is None:
            header_chunk = chunk
            logging.info(
                "Captured header chunk (%d bytes) for STT stream reconnection.",
                len(header_chunk),
            )
          t_recv = time.perf_counter()
          session.total_bytes += len(chunk)
          await audio_queue.put((chunk, t_recv))
          await safe_send_json({"total_bytes": session.total_bytes})
      elif "text" in message:
        # Control messages from the UI.
        logging.info("WebSocket received text message: %s", message["text"])
        if message["text"] == "PING":
          await safe_send_json({"type": "pong", "ws_status": "live"})
        elif message["text"] == "PAUSE":
          await record_system_event("pause", "Session Paused")
          await stop_background_tasks()
        elif message["text"] == "RESUME":
          ensure_session_start_time(session)
          await record_system_event("resume", "Session Resumed")
          # The browser starts a new recorder, so expect a new WebM header.
          header_chunk = None
          stream_counter = 0
          start_background_tasks()
        elif message["text"] == "STOP":
          break
  except fastapi.WebSocketDisconnect:
    logging.info("WebSocket client disconnected.")
  except Exception as e:  # pylint: disable=broad-except
    logging.exception("WebSocket endpoint error: %s", e)
  finally:
    # Detach this socket and stop background work. Session state is kept so the
    # browser can reconnect and resume.
    logging.info(
        "Cleaning up WebSocket session %s. Total bytes received: %d",
        session_id,
        session.total_bytes,
    )
    if session.active_websocket == websocket:
      session.active_websocket = None

    await audio_queue.put(None)
    await stop_background_tasks()


@app.websocket("/ws")
async def websocket_endpoint_legacy(websocket: fastapi.WebSocket):
  """Backward-compatible endpoint that uses a shared "default" session."""
  await websocket_endpoint(websocket, "default")


def main(argv: Sequence[str]) -> None:
  """Serves the FastAPI app with uvicorn on --host:--port."""
  if len(argv) > 1:
    raise absl_app.UsageError("Too many command-line arguments.")
  logging.info(
      "Starting server on %s:%d (open http://localhost:%d in your browser)",
      FLAGS.host,
      FLAGS.port,
      FLAGS.port,
  )
  uvicorn.run(app, host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  # Marked required here, not at definition, so tests can import this module
  # without passing --project.
  flags.mark_flag_as_required("project")
  absl_app.run(main)
