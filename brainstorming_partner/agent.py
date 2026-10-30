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

"""Brainstorming agent module for defining agent roles and requesting insights.

BrainstormAgent sends the live transcript to Gemini on Vertex AI and returns a
structured summary plus an optional suggestion to share with the team.
"""

import asyncio
from collections.abc import Sequence
import json
import os
import ssl
import time
from absl import logging
from google import genai
from google.genai import types
import pydantic


class BrainstormResponse(pydantic.BaseModel):
  """Structured output that Gemini returns for each transcript analysis.

  Attributes:
    summary: Summary of the conversation since the last AI suggestion.
    should_interrupt: Whether the agent wants to interject now.
    interrupt_text: The suggestion to share; empty if not interrupting.
  """

  summary: str
  should_interrupt: bool
  interrupt_text: str


# Default prompt template. analyze_transcript() fills in {history} with earlier
# segment summaries and suggestions, and {transcript} with the latest speech.
DEFAULT_ROLE_PROMPT = """You are an AI brainstorming coworker listening to a
live conversation, actively generating and riffing on ideas alongside
teammates, more of a peer than a guide. The team are collaborating
on a creative project in an abbreviated (90 minute) design sprint. The goal is
to first generate a broad range of ideas, then refine those ideas down to a
select few concepts to develop further.

Your primary goal is to help move the creative process forward with timely and
constructive contributions. Be practical, concise, and follow the best practices
of a design sprint process. Ask thought-provoking questions, offer constructive feedback, 
challenge assumptions, identify gaps, and from time to time, build upon previous ideas.

Your primary goal is to help move the creative process forward with timely and
constructive contributions. Be practical, concise, and follow the best
practices of a human-centered design process. Anchor ideas in reality and things that
are feasible to build and market for Maya, who is 38 years old, works as an
operations manager, works from home 3 days a week, and rents a 2-bedroom apartment.

Previous Conversation History (Summaries & AI Suggestions):
{history}

Current Running Transcript (since last AI suggestion):
{transcript}

Tasks:
1. Provide a concise summary of the current transcript conversation since the last AI suggestion (or since the beginning of the conversation if no AI suggestion has occurred yet).
2. Determine whether you should interrupt the conversation right now (should_interrupt=true/false). Set should_interrupt=true ONLY when you have a valuable creative suggestion, constructive question, or relevant insight worth sharing.
3. If should_interrupt=true, set interrupt_text to your clear, constructive insight or suggestion. If should_interrupt=false, set interrupt_text to an empty string."""


class BrainstormAgent:
  """Agent that analyzes conversation transcripts to provide insightful questions or constructive feedback."""

  def __init__(
      self,
      project: str,
      model_name: str = "gemini-2.5-flash",
      location: str | None = None,
      role_prompt: str = DEFAULT_ROLE_PROMPT,
  ):
    """Initializes the agent and its Vertex AI Gemini client.

    Args:
      project: Google Cloud project ID used for Vertex AI requests and quota.
      model_name: Gemini model ID.
      location: Vertex AI location. If None, it is chosen based on model_name.
      role_prompt: Prompt template with a {transcript} placeholder and,
        optionally, a {history} placeholder.
    """
    self.project = project
    self.model_name = model_name
    if location is None:
      # Automatically set location based on model requirements.
      # Gemini 3.6 models are only supported in the 'global' location on Vertex AI.
      if "3.6" in model_name:
        self.location = "global"
      else:
        self.location = "us-central1"
    else:
      self.location = location
    self.role_prompt = role_prompt
    logging.info(
        "BrainstormAgent initialized: project='%s', model_name='%s',"
        " location='%s'",
        self.project,
        self.model_name,
        self.location,
    )
    # Bill Gemini usage to `project`, as the Speech-to-Text client does, unless
    # GOOGLE_CLOUD_QUOTA_PROJECT is already set. Without a quota project, user
    # credentials can fail with quota or "API not enabled" errors.
    os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", self.project)
    # Some Python builds (e.g. the python.org macOS installer) have no default
    # CA certificates, so HTTPS calls fail with CERTIFICATE_VERIFY_FAILED. Use
    # certifi's CA bundle in that case.
    if (
        not os.environ.get("SSL_CERT_FILE")
        and ssl.get_default_verify_paths().cafile is None
    ):
      try:
        import certifi  # pylint: disable=g-import-not-at-top

        os.environ["SSL_CERT_FILE"] = certifi.where()
      except ImportError:
        pass
    self.client = genai.Client(
        vertexai=True,
        project=self.project,
        location=self.location,
    )

  async def health_check(self) -> float:
    """Sends a short ping request to Gemini API and returns latency_ms."""
    t_start = time.perf_counter()
    await self.client.aio.models.generate_content(
        model=self.model_name,
        contents="Ping",
    )
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000

  async def analyze_transcript(
      self,
      transcript: str,
      history: Sequence[dict[str, str]] | None = None,
  ) -> tuple[BrainstormResponse, float]:
    """Analyzes the transcript and returns (response, latency_ms)."""
    t_start = time.perf_counter()

    # Render earlier segments as numbered summary/suggestion pairs.
    if history:
      history_lines = []
      segment_num = 1
      for item in history:
        s_text = item.get("summary", "").strip()
        i_text = item.get("suggestion", "").strip()
        if not s_text and not i_text:
          continue
        history_lines.append(
            f"Segment {segment_num} Summary: {s_text}\nSegment {segment_num} AI"
            f" Suggestion: {i_text}"
        )
        segment_num += 1
      history_str = (
          "\n\n".join(history_lines)
          if history_lines
          else "None (No previous AI suggestions yet)."
      )
    else:
      history_str = "None (No previous AI suggestions yet)."

    # Custom role prompts may omit the {history} placeholder.
    if "{history}" in self.role_prompt:
      prompt = self.role_prompt.format(
          history=history_str, transcript=transcript
      )
    else:
      prompt = self.role_prompt.format(transcript=transcript)

    # Ask for JSON that matches the BrainstormResponse schema.
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BrainstormResponse,
    )

    logging.info(
        "BrainstormAgent: Sending request to Gemini API | model='%s' |"
        " project='%s' | location='%s' | transcript_len=%d chars",
        self.model_name,
        self.project,
        self.location,
        len(transcript),
    )
    logging.debug(
        "BrainstormAgent: Gemini request config: response_mime_type=%s,"
        " response_schema=%s",
        config.response_mime_type,
        config.response_schema,
    )
    logging.debug("BrainstormAgent: Full prompt sent to Gemini:\n%s", prompt)

    # Retry failed calls with exponential backoff (1s, then 2s).
    max_retries = 3
    base_delay = 1.0
    response = None

    for attempt in range(1, max_retries + 1):
      try:
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        break
      except Exception as err:  # pylint: disable=broad-except
        logging.warning(
            "BrainstormAgent: Gemini API call failed (attempt %d/%d) for model"
            " '%s' (project '%s', location '%s'): %s",
            attempt,
            max_retries,
            self.model_name,
            self.project,
            self.location,
            err,
        )
        if attempt == max_retries:
          logging.exception(
              "BrainstormAgent: All %d attempts failed for model '%s'",
              max_retries,
              self.model_name,
          )
          raise
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    if response is None:
      raise RuntimeError("Failed to obtain response from Gemini API.")

    raw_text = response.text or "{}"
    logging.info(
        "BrainstormAgent: Gemini API response received in %.2f ms |"
        " raw_text_len=%d chars",
        latency_ms,
        len(raw_text),
    )
    logging.debug("BrainstormAgent: Gemini raw response text: %s", raw_text)

    # Accept snake_case or camelCase keys. If the output isn't valid JSON, use
    # the raw text as the summary instead of failing.
    try:
      data_dict = json.loads(raw_text)
      parsed_response = BrainstormResponse(
          summary=data_dict.get("summary", ""),
          should_interrupt=bool(
              data_dict.get(
                  "should_interrupt", data_dict.get("shouldInterrupt", False)
              )
          ),
          interrupt_text=data_dict.get(
              "interrupt_text", data_dict.get("interruptText", "")
          ),
      )
      logging.info(
          "BrainstormAgent: Parsed Gemini response successfully: summary='%s',"
          " should_interrupt=%s, interrupt_text='%s'",
          parsed_response.summary,
          parsed_response.should_interrupt,
          parsed_response.interrupt_text,
      )
    except Exception as parse_err:  # pylint: disable=broad-except
      logging.warning(
          "BrainstormAgent: Failed to parse Gemini JSON output '%s': %s",
          raw_text,
          parse_err,
      )
      parsed_response = BrainstormResponse(
          summary=raw_text,
          should_interrupt=False,
          interrupt_text="",
      )

    return parsed_response, latency_ms
