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

"""Tests for FastAPI server and BrainstormAgent."""

import os
import sys
import tempfile
import unittest
from unittest import mock

# pylint: disable=g-import-not-at-top
# Let `import agent` and `import server` find the sibling modules when this test
# runs outside of Blaze.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from absl.testing import absltest
import agent  # pyrefly: ignore[missing-import]
import fastapi
from fastapi.testclient import TestClient
import server  # pyrefly: ignore[missing-import]

# pylint: enable=g-import-not-at-top


class AgentTest(unittest.IsolatedAsyncioTestCase):
  """Tests BrainstormAgent with a mocked Gemini client."""

  @mock.patch.object(agent.genai, "Client")
  async def test_analyze_transcript(self, mock_genai_client_class):
    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Discussing UI redesign.", "should_interrupt": true,'
        ' "interrupt_text": "Consider dark mode support."}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    brainstorm_agent = agent.BrainstormAgent(
        project="test-project",
        model_name="gemini-2.5-flash",
    )
    resp, latency = await brainstorm_agent.analyze_transcript(
        "[00:01] Let's redesign the dashboard."
    )

    self.assertEqual(resp.summary, "Discussing UI redesign.")
    self.assertTrue(resp.should_interrupt)
    self.assertEqual(resp.interrupt_text, "Consider dark mode support.")
    self.assertGreater(latency, 0.0)

  @mock.patch.object(agent.genai, "Client")
  async def test_analyze_transcript_with_history(self, mock_genai_client_class):
    """Earlier segments are included in the prompt sent to Gemini."""
    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Discussing next step.", "should_interrupt": false,'
        ' "interrupt_text": ""}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    brainstorm_agent = agent.BrainstormAgent(
        project="test-project",
        model_name="gemini-2.5-flash",
    )
    history = [
        {"summary": "Segment 1 summary", "suggestion": "Segment 1 suggestion"}
    ]
    resp, latency = await brainstorm_agent.analyze_transcript(
        "[00:05] What should we do next?", history=history
    )

    self.assertEqual(resp.summary, "Discussing next step.")
    self.assertFalse(resp.should_interrupt)
    call_args = mock_genai_client.aio.models.generate_content.call_args
    prompt_sent = call_args.kwargs.get("contents") or call_args.args[0]
    self.assertIn("Segment 1 Summary: Segment 1 summary", prompt_sent)
    self.assertIn("Segment 1 AI Suggestion: Segment 1 suggestion", prompt_sent)
    self.assertIn(
        "Current Running Transcript (since last AI suggestion):", prompt_sent
    )

  @mock.patch.object(agent.asyncio, "sleep", new_callable=mock.AsyncMock)
  @mock.patch.object(agent.genai, "Client")
  async def test_analyze_transcript_retry_success(
      self, mock_genai_client_class, mock_sleep
  ):
    """A failed Gemini call is retried and the retry's result returned."""
    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Recovered summary.", "should_interrupt": false,'
        ' "interrupt_text": ""}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        side_effect=[
            Exception("500 Internal error encountered"),
            mock_gemini_response,
        ]
    )

    brainstorm_agent = agent.BrainstormAgent(
        project="test-project",
        model_name="gemini-2.5-flash",
    )
    resp, latency = await brainstorm_agent.analyze_transcript(
        "[00:01] Hello world"
    )

    self.assertEqual(resp.summary, "Recovered summary.")
    self.assertEqual(
        mock_genai_client.aio.models.generate_content.call_count, 2
    )

  @mock.patch.object(agent.genai, "Client")
  async def test_health_check(self, mock_genai_client_class):
    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = "Pong"
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    brainstorm_agent = agent.BrainstormAgent(
        project="test-project",
        model_name="gemini-2.5-flash",
    )
    latency = await brainstorm_agent.health_check()
    self.assertGreaterEqual(latency, 0.0)
    mock_genai_client.aio.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash", contents="Ping"
    )


class ServerTest(unittest.TestCase):
  """Tests the HTTP and WebSocket endpoints.

  Tests that need API responses mock the Speech-to-Text and Gemini clients.
  """

  def setUp(self):
    super().setUp()
    # Use a fake project, the V1 STT path, the bundled chime, and short
    # intervals so tests are fast and isolated.
    server.FLAGS.project = "test-project"
    server.FLAGS.speech_model = ""
    server.FLAGS.chime_sound_path = ""
    server.FLAGS.insight_interval_sec = 0.01
    server.FLAGS.interruption_cooldown_sec = 0.0
    server.SESSIONS.clear()
    server.clear_api_status_cache()
    self.client = TestClient(server.app)

  def test_get_index(self):
    response = self.client.get("/", follow_redirects=False)
    self.assertEqual(response.status_code, 303)
    self.assertIn("/session/", response.headers["location"])

    session_url = response.headers["location"]
    response2 = self.client.get(session_url)
    self.assertEqual(response2.status_code, 200)
    self.assertIn("Brainstorming Coworker", response2.text)
    self.assertIn("chimeSound", response2.text)

  def test_get_session_page_nonexistent(self):
    """Unknown session IDs redirect to the root, which starts a new one."""
    response = self.client.get(
        "/session/nonexistent-id", follow_redirects=False
    )
    self.assertEqual(response.status_code, 303)
    self.assertEqual(response.headers["location"], "/")

    response_following = self.client.get(
        "/session/nonexistent-id", follow_redirects=True
    )
    self.assertEqual(response_following.status_code, 200)
    self.assertIn("Brainstorming Coworker", response_following.text)
    self.assertEqual(len(server.SESSIONS), 1)

  def test_index_html_contains_new_session_button_and_logging(self):
    # Setup a session so get_session_page returns index.html
    session_id = "test-session-id"
    server.SESSIONS[session_id] = server.SessionState(session_id)
    response = self.client.get(f"/session/{session_id}")
    self.assertEqual(response.status_code, 200)
    self.assertIn('id="newSessionButton"', response.text)
    self.assertIn("Session ID:", response.text)
    self.assertIn("window.location.href = '/';", response.text)

  def test_index_html_contains_cloud_api_logging_and_reconnection(self):
    session_id = "test-session-id-2"
    server.SESSIONS[session_id] = server.SessionState(session_id)
    response = self.client.get(f"/session/{session_id}")
    self.assertEqual(response.status_code, 200)
    self.assertIn("Speech-to-Text API:", response.text)
    self.assertIn("Brainstorming Partner API:", response.text)
    self.assertIn("updateSttApiState", response.text)
    self.assertIn("updateGeminiApiState", response.text)
    self.assertIn("scheduleWebSocketReconnect", response.text)
    self.assertIn("startHeartbeat", response.text)
    self.assertIn("handleConnectionDrop", response.text)
    self.assertIn("restartAudioStream", response.text)
    self.assertIn("resumeListeningAutomatically", response.text)
    self.assertIn("wasListeningBeforeError", response.text)
    self.assertIn("addEventListener('offline'", response.text)
    self.assertIn("addEventListener('online'", response.text)

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_get_api_status_default(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """Without a Chirp model, /api/status reports the V1 STT model."""
    mock_speech_client = mock_speech_client_class.return_value
    mock_speech_client.recognize = mock.AsyncMock(return_value=mock.MagicMock())

    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = "Pong"
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    response = self.client.get("/api/status")
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertIn("stt", data)
    self.assertEqual(data["stt"]["status"], "live")
    self.assertIn("default (V1)", data["stt"]["model"])

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v2, "SpeechAsyncClient")
  def test_get_api_status_chirp_3(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    server.FLAGS.speech_model = "chirp_3"
    mock_speech_client = mock_speech_client_class.return_value
    mock_speech_client.recognize = mock.AsyncMock(return_value=mock.MagicMock())

    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = "Pong"
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    response = self.client.get("/api/status")
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertIn("stt", data)
    self.assertEqual(data["stt"]["status"], "live")
    self.assertIn("chirp_3", data["stt"]["model"])

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_get_api_status_with_session_id(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    mock_speech_client = mock_speech_client_class.return_value
    mock_speech_client.recognize = mock.AsyncMock(return_value=mock.MagicMock())

    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = "Pong"
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    session_id = "test-status-session"
    session_state = server.SessionState(session_id)
    server.SESSIONS[session_id] = session_state

    response = self.client.get(f"/api/status?session_id={session_id}")
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertIn("session", data)
    self.assertIsNotNone(data["session"])
    self.assertFalse(data["session"]["is_paused"])
    self.assertFalse(data["session"]["stt_task_running"])
    self.assertFalse(data["session"]["gemini_task_running"])

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_get_api_status_caches_live_results(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """Repeated /api/status calls reuse cached live API results while updating session state."""
    mock_speech_client = mock_speech_client_class.return_value
    mock_speech_client.recognize = mock.AsyncMock(return_value=mock.MagicMock())

    mock_genai_client = mock_genai_client_class.return_value
    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = "Pong"
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    session_id = "test-cached-session"
    session_state = server.SessionState(session_id)
    server.SESSIONS[session_id] = session_state

    resp1 = self.client.get(f"/api/status?session_id={session_id}")
    self.assertEqual(resp1.status_code, 200)
    self.assertFalse(resp1.json()["session"]["is_paused"])

    session_state.is_paused = True
    resp2 = self.client.get(f"/api/status?session_id={session_id}")
    self.assertEqual(resp2.status_code, 200)
    self.assertTrue(resp2.json()["session"]["is_paused"])
    self.assertEqual(mock_speech_client.recognize.call_count, 1)
    self.assertEqual(
        mock_genai_client.aio.models.generate_content.call_count, 1
    )

  def test_flags_configuration(self):
    server.FLAGS.speech_model = ""
    self.assertEqual(server.FLAGS.speech_model, "")
    self.assertEqual(server.FLAGS.speech_location, "us")

  def test_chime_sound_endpoint(self):
    response = self.client.get("/audio/notification.ogg")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers["content-type"], "audio/ogg")

  def test_chime_sound_custom_flag_path(self):
    with tempfile.NamedTemporaryFile(suffix=".ogg") as tmp:
      tmp.write(b"OggS-custom-chime")
      tmp.flush()
      server.FLAGS.chime_sound_path = tmp.name
      response = self.client.get("/audio/notification.ogg")
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.headers["content-type"], "audio/ogg")
      self.assertEqual(response.content, b"OggS-custom-chime")

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_stream_size_and_transcribe(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """Audio bytes are counted and STT results reach the browser."""
    mock_speech_client = mock_speech_client_class.return_value

    async def mock_streaming_recognize(requests):
      async def response_generator():
        # Yield interim result
        mock_response_1 = mock.MagicMock()
        mock_result_1 = mock.MagicMock()
        mock_alt_1 = mock.MagicMock()
        mock_alt_1.transcript = "hello"
        mock_alt_1.words = []
        mock_result_1.alternatives = [mock_alt_1]
        mock_result_1.is_final = False
        mock_response_1.results = [mock_result_1]
        yield mock_response_1

        # Consume requests to simulate waiting for audio stream
        async for _ in requests:
          pass

        # Yield final result
        mock_response_2 = mock.MagicMock()
        mock_result_2 = mock.MagicMock()
        mock_alt_2 = mock.MagicMock()
        mock_alt_2.transcript = "hello world"
        mock_alt_2.words = []
        mock_result_2.alternatives = [mock_alt_2]
        mock_result_2.is_final = True
        mock_response_2.results = [mock_result_2]
        yield mock_response_2

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    with self.client.websocket_connect("/ws") as websocket:
      initial_msg = websocket.receive_json()
      self.assertIn("transcript", initial_msg)
      self.assertIn("hello", initial_msg["transcript"])
      self.assertFalse(initial_msg["is_final"])

      websocket.send_bytes(b"1234567890")

      bytes_msg = websocket.receive_json()
      self.assertEqual(bytes_msg["total_bytes"], 10)

      websocket.close()

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_stt_error(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """Repeated STT failures send retry warnings, then a final error."""
    mock_speech_client = mock_speech_client_class.return_value
    # Every STT attempt fails, so the tracebacks this test logs are expected.
    mock_speech_client.streaming_recognize.side_effect = Exception(
        "API connection failed"
    )

    with self.client.websocket_connect("/ws") as websocket:
      websocket.send_bytes(b"1234567890")

      warning_msgs = []
      error_msg = None
      while True:
        try:
          data = websocket.receive_json()
          if "stt_warning" in data:
            warning_msgs.append(data["stt_warning"])
          if "error" in data:
            error_msg = data
            break
        except Exception:
          break

      self.assertGreater(len(warning_msgs), 0)
      self.assertIsNotNone(error_msg)
      self.assertIn("API connection failed", error_msg["error"])
      websocket.close()

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_gemini_insight(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """A Gemini suggestion reaches the browser and the transcript."""
    mock_speech_client = mock_speech_client_class.return_value
    mock_genai_client = mock_genai_client_class.return_value

    async def mock_streaming_recognize(requests):
      async def response_generator():
        mock_response = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_alt = mock.MagicMock()
        mock_alt.transcript = "Testing Gemini coworker integration"
        mock_alt.words = []
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_response.results = [mock_result]
        yield mock_response
        async for _ in requests:
          pass

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Team is testing Gemini integration.", "should_interrupt":'
        ' true, "interrupt_text": "Keep up the good work."}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    with self.client.websocket_connect("/ws") as websocket:
      stt_msg = websocket.receive_json()
      self.assertIn(
          "Testing Gemini coworker integration", stt_msg["transcript"]
      )

      gemini_msg = None
      while True:
        try:
          data = websocket.receive_json()
          if "gemini_summary" in data:
            gemini_msg = data
            break
        except Exception:
          break

      self.assertIsNotNone(gemini_msg)
      self.assertEqual(
          gemini_msg["gemini_summary"], "Team is testing Gemini integration."
      )
      self.assertTrue(gemini_msg["should_interrupt"])
      self.assertEqual(gemini_msg["interrupt_text"], "Keep up the good work.")
      self.assertIn("gemini_latency_ms", gemini_msg)
      self.assertIn(
          "AI Coworker: Keep up the good work.", gemini_msg["transcript"]
      )

      websocket.close()

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_interruption_cooldown(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """The first suggestion is delivered even with a long cooldown."""
    server.FLAGS.interruption_cooldown_sec = 60.0
    mock_speech_client = mock_speech_client_class.return_value
    mock_genai_client = mock_genai_client_class.return_value

    async def mock_streaming_recognize(requests):
      async def response_generator():
        mock_response = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_alt = mock.MagicMock()
        mock_alt.transcript = "Testing cooldown"
        mock_alt.words = []
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_response.results = [mock_result]
        yield mock_response
        async for _ in requests:
          pass

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Testing cooldown.", "should_interrupt": true,'
        ' "interrupt_text": "Suggestion 1"}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    with self.client.websocket_connect("/ws") as websocket:
      websocket.receive_json()

      gemini_msg1 = None
      while True:
        try:
          data = websocket.receive_json()
          if "gemini_summary" in data:
            gemini_msg1 = data
            break
        except Exception:
          break

      self.assertIsNotNone(gemini_msg1)
      self.assertTrue(gemini_msg1["should_interrupt"])
      self.assertEqual(gemini_msg1["interrupt_text"], "Suggestion 1")

      websocket.close()

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_history_and_segment_slicing(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """An interjection closes the current segment and adds it to history."""
    mock_speech_client = mock_speech_client_class.return_value
    mock_genai_client = mock_genai_client_class.return_value

    async def mock_streaming_recognize(requests):
      async def response_generator():
        mock_response = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_alt = mock.MagicMock()
        mock_alt.transcript = "First segment text"
        mock_alt.words = []
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_response.results = [mock_result]
        yield mock_response
        async for _ in requests:
          pass

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    mock_gemini_response = mock.MagicMock()
    mock_gemini_response.text = (
        '{"summary": "Summary for segment 1.", "should_interrupt":'
        ' true, "interrupt_text": "Suggestion 1"}'
    )
    mock_genai_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_gemini_response
    )

    with self.client.websocket_connect("/ws") as websocket:
      stt_msg = websocket.receive_json()
      self.assertIn("First segment text", stt_msg["transcript"])

      gemini_msg = None
      while True:
        try:
          data = websocket.receive_json()
          if "history" in data and len(data["history"]) > 0:
            gemini_msg = data
            break
        except Exception:
          break

      self.assertIsNotNone(gemini_msg)
      self.assertEqual(len(gemini_msg["history"]), 1)
      self.assertEqual(
          gemini_msg["history"][0]["summary"], "Summary for segment 1."
      )
      self.assertEqual(gemini_msg["history"][0]["suggestion"], "Suggestion 1")
      self.assertIn("timestamp_range", gemini_msg["history"][0])
      websocket.close()

  def test_websocket_session_resume(self):
    """Reconnecting replays the session state and marks the reload."""
    # Setup some pre-existing session state
    session_id = "session-123"
    session_state = server.SessionState(session_id)
    session_state.latest_transcript = "[00:01] Hello world"
    session_state.history = [
        {"summary": "Greeting", "suggestion": "Say hi back"}
    ]
    server.SESSIONS[session_id] = session_state

    # Connect to the session-specific websocket
    with self.client.websocket_connect(f"/ws/{session_id}") as websocket:
      # The first message received should be the resumed session state
      msg = websocket.receive_json()
      self.assertEqual(
          msg["transcript"], "[00:01] Hello world\n[00:00] Session Reloaded"
      )
      # There should be the original history item PLUS the reloaded event
      self.assertEqual(len(msg["history"]), 2)
      self.assertEqual(msg["history"][0]["summary"], "Greeting")
      self.assertEqual(msg["history"][1]["system_event"], "reload")
      self.assertIn("Session Reloaded", msg["history"][1]["text"])
      websocket.close()

  def test_websocket_session_takeover(self):
    """A second connection to a session closes the first one."""
    session_id = "session-456"
    with self.client.websocket_connect(f"/ws/{session_id}") as ws1:
      # A second connection takes over
      with self.client.websocket_connect(f"/ws/{session_id}") as ws2:
        # The first socket should get disconnected/closed
        for _ in range(10):
          try:
            ws1.receive_json()
          except fastapi.WebSocketDisconnect as ctx:
            self.assertEqual(ctx.code, 1000)
            self.assertEqual(ctx.reason, "Resumed in another connection")
            break
        else:
          self.fail("WebSocketDisconnect not raised")
        ws2.close()

  def test_websocket_pause_and_resume(self):
    session_id = "session-pause-resume"
    with self.client.websocket_connect(f"/ws/{session_id}") as websocket:
      websocket.send_text("PAUSE")
      msg_pause = None
      for _ in range(10):
        data = websocket.receive_json()
        if data.get("system_event") == "pause":
          msg_pause = data
          break
      self.assertIsNotNone(msg_pause)
      self.assertTrue(server.SESSIONS[session_id].is_paused)

      websocket.send_text("RESUME")
      msg_resume = None
      for _ in range(10):
        data = websocket.receive_json()
        if data.get("system_event") == "resume":
          msg_resume = data
          break
      self.assertIsNotNone(msg_resume)
      self.assertFalse(server.SESSIONS[session_id].is_paused)
      websocket.close()

  def test_is_idle_or_stream_timeout(self):
    self.assertTrue(
        server.is_idle_or_stream_timeout(
            Exception(
                "409 Stream timed out after receiving no more client requests."
                ' [detail: "[ORIGINAL ERROR] generic::aborted: Stream timed out'
                ' after receiving no more client requests."]'
            )
        )
    )
    self.assertTrue(
        server.is_idle_or_stream_timeout(
            Exception(
                "generic::aborted: Stream timed out after receiving no more"
                " client requests."
            )
        )
    )
    self.assertTrue(
        server.is_idle_or_stream_timeout(
            Exception(
                "Exceeded maximum allowed stream duration of 305 seconds."
            )
        )
    )
    self.assertTrue(
        server.is_idle_or_stream_timeout(
            Exception("Out of range: audio data duration exceeded.")
        )
    )
    self.assertFalse(
        server.is_idle_or_stream_timeout(
            Exception("403 Permission denied on resource.")
        )
    )
    self.assertFalse(
        server.is_idle_or_stream_timeout(Exception("API connection failed."))
    )

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_idle_timeout_handling(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """An STT idle timeout reconnects quietly instead of erroring."""
    mock_speech_client = mock_speech_client_class.return_value
    call_count = 0

    async def mock_streaming_recognize(requests):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        # First attempt: simulate idle timeout
        raise Exception(
            "409 Stream timed out after receiving no more client requests."
        )

      # Second attempt: succeed with transcript when audio arrives
      async def response_generator():
        mock_response = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_alt = mock.MagicMock()
        mock_alt.transcript = "speech after idle"
        mock_alt.words = []
        mock_result.alternatives = [mock_alt]
        mock_result.is_final = True
        mock_response.results = [mock_result]
        yield mock_response
        async for _ in requests:
          pass

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    with self.client.websocket_connect("/ws") as websocket:
      # Send a ping to verify connection is alive
      websocket.send_text("PING")
      pong_msg = websocket.receive_json()
      self.assertEqual(pong_msg.get("type"), "pong")

      # Send audio bytes after being idle
      websocket.send_bytes(b"audio-data")

      received_transcript = None
      for _ in range(10):
        try:
          data = websocket.receive_json()
          if "stt_warning" in data:
            self.fail(f"Unexpected stt_warning received on idle: {data}")
          if "error" in data:
            self.fail(f"Unexpected error received on idle: {data}")
          if "transcript" in data and "speech after idle" in data["transcript"]:
            received_transcript = data["transcript"]
            break
        except Exception:
          break

      self.assertIsNotNone(received_transcript)
      self.assertIn("speech after idle", received_transcript)
      websocket.close()

  @mock.patch.object(agent.genai, "Client")
  @mock.patch.object(server.speech_v1, "SpeechAsyncClient")
  def test_websocket_silence_recovery(
      self, mock_speech_client_class, mock_genai_client_class
  ):
    """Transcription resumes after the STT stream times out in silence."""
    mock_speech_client = mock_speech_client_class.return_value
    call_count = 0

    async def mock_streaming_recognize(requests):
      nonlocal call_count
      call_count += 1
      current_call = call_count

      async def response_generator():
        if current_call == 1:
          # First utterance
          mock_response = mock.MagicMock()
          mock_result = mock.MagicMock()
          mock_alt = mock.MagicMock()
          mock_alt.transcript = "first utterance"
          mock_alt.words = []
          mock_result.alternatives = [mock_alt]
          mock_result.is_final = True
          mock_response.results = [mock_result]
          yield mock_response
          # Simulate idle timeout at end of silence
          raise Exception(
              "generic::aborted: Stream timed out after receiving no more"
              " client requests."
          )
        else:
          # Utterance after silence
          mock_response = mock.MagicMock()
          mock_result = mock.MagicMock()
          mock_alt = mock.MagicMock()
          mock_alt.transcript = "second utterance"
          mock_alt.words = []
          mock_result.alternatives = [mock_alt]
          mock_result.is_final = True
          mock_response.results = [mock_result]
          yield mock_response
          async for _ in requests:
            pass

      return response_generator()

    mock_speech_client.streaming_recognize.side_effect = (
        mock_streaming_recognize
    )

    with self.client.websocket_connect("/ws") as websocket:
      # Send first audio chunk
      websocket.send_bytes(b"chunk-1")

      first_transcript = None
      for _ in range(10):
        data = websocket.receive_json()
        if "transcript" in data and "first utterance" in data["transcript"]:
          first_transcript = data["transcript"]
          break

      self.assertIsNotNone(first_transcript)

      # Send second audio chunk after silence
      websocket.send_bytes(b"chunk-2")

      second_transcript = None
      for _ in range(10):
        data = websocket.receive_json()
        if "transcript" in data and "second utterance" in data["transcript"]:
          second_transcript = data["transcript"]
          break

      self.assertIsNotNone(second_transcript)
      self.assertIn("second utterance", second_transcript)
      websocket.close()


if __name__ == "__main__":
  absltest.main()
