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

"""Module 05: Initial Events Discovery via LLM

Initial event discovery using LLM-guided web search.
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from reprlib import Repr
from typing import Dict, List, Optional

from google import genai
from google.genai import types
import pandas as pd

from .llm_retry_handler import llm_retry_on_429
from .llm_retry_handler import (
    PendingRetryProviderError,
    ProviderSemanticError,
    is_llm_rate_limit_error,
    is_provider_timeout_error,
)

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LLM_SEARCH_HTTP_TIMEOUT_MS = int(
    os.getenv('GEMINI_SEARCH_HTTP_TIMEOUT_MS', '60000')
)


def _create_gemini_client(api_key):
  original_google_api_key = os.environ.pop('GOOGLE_API_KEY', None)
  try:
    http_options = types.HttpOptions(timeout=LLM_SEARCH_HTTP_TIMEOUT_MS)
    return genai.Client(api_key=api_key, http_options=http_options)
  finally:
    if original_google_api_key is not None:
      os.environ['GOOGLE_API_KEY'] = original_google_api_key


EVENT_TYPE_CLASSIFICATION_TEXT = """3.  **Event Type Classification:** Please classify the event into the following types based on its certainty and timing.
    *   `Scheduled Event`: A high-certainty event that has been officially announced to occur at a future date.
    *   `Predictive Information`: A lower-certainty piece of information about the future, such as an analyst forecast, a target price change, or a credible rumor.
    *   `Contemporaneous Event`: An event that occurs at the same time it is announced, often unexpected.
    *   `Retrospective Report`: An analysis or report about an event or period that has already passed."""


@llm_retry_on_429(max_retries=5)
async def get_initial_events_via_llm(
    keyword,
    time_range,
    domain,
    geography_str,
    config,
):
  """LLM

  Args:
      keyword:
      time_range:  {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
      domain:
      geography_str:
      config:

  Returns:
      List of events with structure:
      {
          "event_id": "unique_hash",
          "event_summary": "Event description",
          "announcement_date": "YYYY-MM-DD",
          "occurrence_date": "YYYY-MM-DD",

          "event_type": "Instantaneous|Anticipated|Protracted|Retrospective",
          "source_urls": ["url1", "url2"],
          "confidence_score": 0.8
      }
  """
  logger.info(
      f"Starting initial event discovery for keyword '{keyword}' in domain"
      f" '{domain}'."
  )

  try:

    api_key = os.environ.get('GEMINI_API_KEY') or GEMINI_API_KEY
    if not api_key:
      logger.error('GEMINI_API_KEY not found')
      return []

    client = _create_gemini_client(api_key)

    model_name = config.get('llm_models', {}).get(
        'initial_event_acquisition', 'gemini-3.5-flash'
    )
    logger.info(
        f'Configured initial event acquisition model setting: '
        f"{config.get('llm_models', {}).get('initial_event_acquisition', 'not configured')}"
    )
    logger.info(f'Using model for initial event acquisition: {model_name}')

    start_date = time_range['start']
    end_date = time_range['end']

    use_geography_in_prompt = (
        config.get('dual_channel_discovery', {})
        .get('llm_channel', {})
        .get('use_geography_in_prompt', True)
    )

    if use_geography_in_prompt:
      prompt = f"""
You are a professional research analyst specializing in the {domain} field. Your task is to use your web search capabilities to identify and structure significant events related to '{keyword}' in {geography_str} that occurred between {start_date} and {end_date}.

**Key Guidelines:**

1.  **Source of Information:** Please base your responses on the information retrieved from your web search and general common sense. It is important to avoid relying on internal knowledge or generating speculative details (hallucinations).

2.  **Date Extraction Principles:** It is helpful to distinguish between two key types of dates. If a date cannot be found from the sources, please use `null`.
    *   `announcement_date`: The date when the news about the event was **published or first announced**. For example, if a news article from **2024-01-10** announces an upcoming product launch.
    *   `occurrence_date`: The date when the event **actually took place or is scheduled to take place**. For example, if the product launch mentioned above happens on **2024-02-02**.

{EVENT_TYPE_CLASSIFICATION_TEXT}

4.  **Geographic Focus:** Please focus on events within the {geography_str} region.

5.  **Source Verifiability:** Each event should be supported by at least one verifiable, high-quality URL.

**Output Format:**

Please provide your response as a JSON array of event objects. Each object in the array should conform to the following structure. If any field's value cannot be determined from the sources, use `null`.

```json
[
            {{
    "event_summary": "A concise, factual description of the event.",
    "announcement_date": "YYYY-MM-DD or null",
    "occurrence_date": "YYYY-MM-DD or null",
    "event_type": "Scheduled Event|Predictive Information|Contemporaneous Event|Retrospective Report or Mixed (such as Scheduled Event+Predictive Information) or  null",
              "source_urls": ["url1", "url2"],
              "confidence_score": 0.8
            }}
]
```

Please ensure the entire response is only the valid JSON array, without any surrounding text or explanations.
            """
    else:
      prompt = f"""
You are a professional research analyst specializing in the {domain} field. Your task is to use your web search capabilities to identify and structure significant events related to '{keyword}' that occurred between {start_date} and {end_date}.

**Key Guidelines:**

1.  **Source of Information:** Please base your responses on the information retrieved from your web search and general common sense. It is important to avoid relying on internal knowledge or generating speculative details (hallucinations).

2.  **Date Extraction Principles:** It is helpful to distinguish between two key types of dates. If a date cannot be found from the sources, please use `null`.
    *   `announcement_date`: The date when the news about the event was **published or first announced**. For example, if a news article from **2024-01-10** announces an upcoming product launch.
    *   `occurrence_date`: The date when the event **actually took place or is scheduled to take place**. For example, if the product launch mentioned above happens on **2024-02-02**.

{EVENT_TYPE_CLASSIFICATION_TEXT}

4.  **Source Verifiability:** Each event should be supported by at least one verifiable, high-quality URL.

**Output Format:**

Please provide your response as a JSON array of event objects. Each object in the array should conform to the following structure. If any field's value cannot be determined from the sources, use `null`.

```json
[
            {{
    "event_summary": "A concise, factual description of the event.",
    "announcement_date": "YYYY-MM-DD or null",
    "occurrence_date": "YYYY-MM-DD or null",
    "event_type": "Scheduled Event|Predictive Information|Contemporaneous Event|Retrospective Report or Mixed (such as Scheduled Event+Predictive Information) or  null",
              "source_urls": ["url1", "url2"],
              "confidence_score": 0.8
            }}
]
```

Please ensure the entire response is only the valid JSON array, without any surrounding text or explanations.
            """

    contents = [
        types.Content(role='user', parts=[types.Part.from_text(text=prompt)])
    ]

    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    generate_content_config = types.GenerateContentConfig(
        max_output_tokens=65000, tools=tools
    )

    logger.info('Sending initial event discovery request to the LLM.')
    response_chunks = []

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
      if chunk.text:
        response_chunks.append(chunk.text)

    full_response = ''.join(response_chunks)
    logger.info(
        f'Received initial event discovery response with {len(full_response)}'
        ' characters.'
    )

    try:

      json_start = full_response.find('[')
      json_end = full_response.rfind(']') + 1

      if json_start == -1 or json_end == 0:
        logger.warning(
            'No JSON array was found in the initial event discovery response.'
        )
        return []

      json_text = full_response[json_start:json_end]
      events_data = json.loads(json_text)

      if not isinstance(events_data, list):
        logger.warning('No additional details were provided for this stage.')
        return []

      processed_events = []
      for i, event in enumerate(events_data):
        try:

          event_content = (
              f"{event.get('event_summary', '')}{event.get('announcement_date', '')}"
          )
          event_id = hashlib.md5(event_content.encode()).hexdigest()[:12]

          processed_event = {
              'event_id': event_id,
              'event_summary': event.get('event_summary', ''),
              'announcement_date': event.get('announcement_date'),
              'occurrence_date': event.get('occurrence_date'),
              'event_type': event.get('event_type', 'Instantaneous'),
              'source_urls': event.get('source_urls', []),
              'confidence_score': float(event.get('confidence_score', 0.8)),
          }

          if (
              processed_event['event_summary']
              and processed_event['announcement_date']
          ):
            processed_events.append(processed_event)
            logger.info(
                f'Accepted initial event {i+1}:'
                f" {processed_event['event_summary'][:50]}..."
            )
          else:
            logger.warning(
                f'Discarded candidate initial event {i+1} because required'
                ' fields were missing.'
            )

        except Exception as e:
          logger.warning(
              f'Failed to normalize candidate initial event {i+1}: {str(e)}'
          )
          continue

      logger.info(
          f'Initial event discovery produced {len(processed_events)} valid'
          ' events.'
      )
      logger.info(f'Normalized initial events: {processed_events}')
      return processed_events

    except json.JSONDecodeError as e:
      logger.error(f'Failed to parse initial event discovery JSON: {str(e)}')
      logger.debug(
          'Initial event discovery raw response preview:'
          f' {full_response[:1000]}...'
      )
      return []

  except ProviderSemanticError:
    raise
  except Exception as e:
    if is_provider_timeout_error(str(e)):
      raise PendingRetryProviderError(
          'gemini_search', 'gemini_search_timeout', str(e)
      ) from e
    if is_llm_rate_limit_error(str(e)):
      raise e
    logger.error(f'Initial event processing failed: {str(e)}')
    return []


def validate_event_data(event):
  """Args:

      event:

  Returns:
      bool:
  """
  required_fields = ['event_summary', 'announcement_date']

  for field in required_fields:
    if not event.get(field):
      return False

  try:
    if event.get('announcement_date'):
      datetime.strptime(event['announcement_date'], '%Y-%m-%d')
    if event.get('occurrence_date'):
      datetime.strptime(event['occurrence_date'], '%Y-%m-%d')

  except (ValueError, TypeError):
    return False

  return True


async def get_initial_events_with_iterations(
    keyword,
    time_range,
    domain,
    geography_str,
    config,
    reference_peaks = None,
):
  """Run iterative initial-event discovery until coverage or count limits are met.

  Args:
      keyword: Target keyword for event discovery.
      time_range: Dictionary with start and end dates.
      domain: Domain label used in prompts.
      geography_str: Geography string used in prompts.
      config: Runtime configuration dictionary.
      reference_peaks: Optional reference peaks for coverage checks.

  Returns:
      A list of discovered initial events.
  """

  max_iterations = config.get('event_sourcing', {}).get(
      'initial_event_iterations', 3
  )

  max_initial_events = config.get('event_sourcing', {}).get(
      'max_initial_events', 100
  )

  logger.info(
      f'Initial discovery limits: max_iterations={max_iterations},'
      f' max_initial_events={max_initial_events}'
  )

  all_events = await get_initial_events_via_llm(
      keyword, time_range, domain, geography_str, config
  )
  logger.info(f'Initial discovery iteration returned {len(all_events)} events.')

  if max_iterations <= 1:

    if len(all_events) > max_initial_events:
      logger.info(
          f'Current event count: {len(all_events)} / {max_initial_events}'
      )
      all_events = all_events[:max_initial_events]
    return all_events

  for iteration in range(2, max_iterations + 1):
    logger.info(f'Starting discovery iteration {iteration}.')

    event_config = config.get('event_sourcing', {})
    if (
        event_config.get('coverage_early_stop_enabled', False)
        and reference_peaks
        and len(all_events) > 0
    ):
      current_coverage = calculate_coverage_rate(
          all_events, reference_peaks, config
      )
      early_stop_threshold = event_config.get(
          'coverage_early_stop_threshold', 0.95
      )
      logger.info(
          f'Coverage early-stop check before iteration {iteration}:'
          f' coverage={current_coverage:.1%},'
          f' threshold={early_stop_threshold:.1%}'
      )
      if current_coverage >= early_stop_threshold:
        logger.info(
            f'Coverage early-stop triggered before iteration {iteration}: '
            f'{current_coverage:.1%} >= {early_stop_threshold:.1%}'
        )
        break
    elif event_config.get('coverage_early_stop_enabled', False):
      logger.info(
          f'Coverage early-stop check before iteration {iteration}: not'
          ' applicable because no reference peaks or no current events are'
          ' available.'
      )

    new_events = await _get_incremental_events(
        keyword, time_range, domain, geography_str, config, all_events
    )

    logger.info(
        f'Discovery iteration {iteration} returned {len(new_events)} new'
        ' events.'
    )
    logger.info(
        f'Discovery iteration {iteration} normalized events: {new_events}'
    )

    if len(new_events) == 0:
      logger.info(f'Starting discovery iteration {iteration}.')
      break

    all_events.extend(new_events)

    if len(all_events) >= max_initial_events:
      logger.info(
          f'Stopping early because the event cap was reached ({len(all_events)}'
          f' >= {max_initial_events}).'
      )
      break

  if len(all_events) > max_initial_events:
    logger.info(
        f'Truncating event list from {len(all_events)} to the configured cap of'
        f' {max_initial_events}.'
    )

    if reference_peaks:
      coverage_before = calculate_coverage_rate(
          all_events, reference_peaks, config
      )
      coverage_after = calculate_coverage_rate(
          all_events[:max_initial_events], reference_peaks, config
      )
      min_coverage = config.get('event_sourcing', {}).get(
          'coverage_min_acceptable', 0.90
      )

      logger.info(f'Coverage before truncation: {coverage_before:.1%}')
      logger.info(f'Coverage after truncation: {coverage_after:.1%}')

      if coverage_after < min_coverage:
        logger.info(
            f'Coverage remains below threshold ({coverage_after:.1%} <'
            f' {min_coverage:.1%}); continuing iterative discovery.'
        )
      else:
        logger.info(f'Coverage target satisfied at {coverage_after:.1%}.')
        all_events = all_events[:max_initial_events]
    else:

      logger.info(
          f'Truncating event list from {len(all_events)} to the configured cap'
          f' of {max_initial_events}.'
      )
      all_events = all_events[:max_initial_events]

  logger.info(f'Current event count: {len(all_events)}')

  if reference_peaks and len(all_events) > 0:
    final_coverage = calculate_coverage_rate(
        all_events, reference_peaks, config
    )
    logger.info(
        f'Final coverage summary: {final_coverage:.1%}'
        f" ({len([p for p in reference_peaks if any(abs((p['peak_date'] - ed).days) <= config.get('event_sourcing', {}).get('coverage_time_window_days', 5) for event in all_events for ed in extract_event_dates(event))])}"
        f' / {len(reference_peaks)})'
    )
  elif reference_peaks:
    logger.info(f'Final coverage summary: 0.0% (0 / {len(reference_peaks)})')

  return all_events


@llm_retry_on_429(max_retries=5)
async def _get_incremental_events(
    keyword,
    time_range,
    domain,
    geography_str,
    config,
    existing_events,
):
  """Request an additional discovery round while excluding already known events.

  Args:
      existing_events: Events already collected in previous rounds.

  Returns:
      A list of newly discovered events.
  """
  logger.info(f'Existing event seed count: {len(existing_events)}')

  try:

    api_key = os.environ.get('GEMINI_API_KEY') or GEMINI_API_KEY
    if not api_key:
      logger.error('GEMINI_API_KEY not found')
      return []

    client = _create_gemini_client(api_key)

    model_name = config.get('llm_models', {}).get(
        'initial_event_acquisition', 'gemini-3.5-flash'
    )
    logger.info(f'Model for incremental discovery: {model_name}')

    known_events_text = '\n'.join([
        f"{i+1}. {event['event_summary']} -"
        f" {event.get('announcement_date', 'Unknown date')}"
        for i, event in enumerate(existing_events)
    ])

    start_date = time_range['start']
    end_date = time_range['end']

    use_geography_in_prompt = (
        config.get('dual_channel_discovery', {})
        .get('llm_channel', {})
        .get('use_geography_in_prompt', True)
    )

    if use_geography_in_prompt:
      prompt = f"""
You are a professional research analyst specializing in the {domain} field. Your task is to use your web search capabilities to identify and structure significant events related to '{keyword}' in {geography_str} that occurred between {start_date} and {end_date}.

**CRITICAL: We have already discovered these events. DO NOT repeat or find similar events:**

{known_events_text}

**Your task: Find OTHER COMPLETELY DIFFERENT events that are NOT listed above and NOT similar to them.**

**ABSOLUTE REQUIREMENT: AVOID DUPLICATES**
- Do NOT find events that are the same as or similar to the ones already discovered
- Focus on finding DIFFERENT types of events, DIFFERENT dates, DIFFERENT aspects
- Look for events from different categories, different stakeholders, or different time periods within the range
- If you cannot find genuinely different events, return an empty array []

**Key Guidelines:**

1.  **Source of Information:** Please base your responses on the information retrieved from your web search and general common sense. It is important to avoid relying on internal knowledge or generating speculative details (hallucinations).

2.  **Date Extraction Principles:** It is helpful to distinguish between two key types of dates. If a date cannot be found from the sources, please use `null`.
    *   `announcement_date`: The date when the news about the event was **published or first announced**. For example, if a news article from **2024-01-10** announces an upcoming product launch.
    *   `occurrence_date`: The date when the event **actually took place or is scheduled to take place**. For example, if the product launch mentioned above happens on **2024-02-02**.

{EVENT_TYPE_CLASSIFICATION_TEXT}

4.  **Geographic Focus:** Please focus on events within the {geography_str} region.

5.  **Source Verifiability:** Each event should be supported by at least one verifiable, high-quality URL.

**Output Format:**

Please provide your response as a JSON array of event objects. Each object in the array should conform to the following structure. If any field's value cannot be determined from the sources, use `null`.

```json
[
            {{
    "event_summary": "A concise, factual description of the event.",
    "announcement_date": "YYYY-MM-DD or null",
    "occurrence_date": "YYYY-MM-DD or null",
    "event_type": "Scheduled Event|Predictive Information|Contemporaneous Event|Retrospective Report or Mixed (such as Scheduled Event+Predictive Information) or  null",
              "source_urls": ["url1", "url2"],
              "confidence_score": 0.8
            }}
]
```

Please ensure the entire response is only the valid JSON array, without any surrounding text or explanations.
            """
    else:
      prompt = f"""
You are a professional research analyst specializing in the {domain} field. Your task is to use your web search capabilities to identify and structure significant events related to '{keyword}' that occurred between {start_date} and {end_date}.

**CRITICAL: We have already discovered these events. DO NOT repeat or find similar events:**

{known_events_text}

**Your task: Find OTHER COMPLETELY DIFFERENT events that are NOT listed above and NOT similar to them.**

**ABSOLUTE REQUIREMENT: AVOID DUPLICATES**
- Do NOT find events that are the same as or similar to the ones already discovered
- Focus on finding DIFFERENT types of events, DIFFERENT dates, DIFFERENT aspects
- Look for events from different categories, different stakeholders, or different time periods within the range
- If you cannot find genuinely different events, return an empty array []

**Key Guidelines:**

1.  **Source of Information:** Please base your responses on the information retrieved from your web search and general common sense. It is important to avoid relying on internal knowledge or generating speculative details (hallucinations).

2.  **Date Extraction Principles:** It is helpful to distinguish between two key types of dates. If a date cannot be found from the sources, please use `null`.
    *   `announcement_date`: The date when the news about the event was **published or first announced**. For example, if a news article from **2024-01-10** announces an upcoming product launch.
    *   `occurrence_date`: The date when the event **actually took place or is scheduled to take place**. For example, if the product launch mentioned above happens on **2024-02-02**.

{EVENT_TYPE_CLASSIFICATION_TEXT}

4.  **Source Verifiability:** Each event should be supported by at least one verifiable, high-quality URL.

**Output Format:**

Please provide your response as a JSON array of event objects. Each object in the array should conform to the following structure. If any field's value cannot be determined from the sources, use `null`.

```json
[
            {{
    "event_summary": "A concise, factual description of the event.",
    "announcement_date": "YYYY-MM-DD or null",
    "occurrence_date": "YYYY-MM-DD or null",
    "event_type": "Scheduled Event|Predictive Information|Contemporaneous Event|Retrospective Report or Mixed (such as Scheduled Event+Predictive Information) or  null",
              "source_urls": ["url1", "url2"],
              "confidence_score": 0.8
            }}
]
```

Please ensure the entire response is only the valid JSON array, without any surrounding text or explanations.
            """

    contents = [
        types.Content(role='user', parts=[types.Part.from_text(text=prompt)])
    ]

    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    generate_content_config = types.GenerateContentConfig(
        max_output_tokens=65000, tools=tools
    )

    logger.info(
        'Sending coverage-supervised event discovery request to the LLM.'
    )
    response_chunks = []

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
      if chunk.text:
        response_chunks.append(chunk.text)

    full_response = ''.join(response_chunks)
    logger.info(
        'Received coverage-supervised discovery response with'
        f' {len(full_response)} characters.'
    )

    try:

      json_start = full_response.find('[')
      json_end = full_response.rfind(']') + 1

      if json_start == -1 or json_end == 0:
        logger.warning(
            'No JSON array was found in the incremental discovery response.'
        )
        return []

      json_text = full_response[json_start:json_end]
      events_data = json.loads(json_text)

      if not isinstance(events_data, list):
        logger.warning('No additional details were provided for this stage.')
        return []

      processed_events = []
      for i, event in enumerate(events_data):
        try:

          event_content = (
              f"{event.get('event_summary', '')}{event.get('announcement_date', '')}"
          )
          event_id = hashlib.md5(event_content.encode()).hexdigest()[:12]

          processed_event = {
              'event_id': event_id,
              'event_summary': event.get('event_summary', ''),
              'announcement_date': event.get('announcement_date'),
              'occurrence_date': event.get('occurrence_date'),
              'event_type': event.get('event_type', 'Instantaneous'),
              'source_urls': event.get('source_urls', []),
              'confidence_score': float(event.get('confidence_score', 0.8)),
          }

          if (
              processed_event['event_summary']
              and processed_event['announcement_date']
          ):
            processed_events.append(processed_event)
            logger.info(
                f'Accepted incremental event {i+1}:'
                f" {processed_event['event_summary'][:50]}..."
            )
          else:
            logger.warning(
                f'Discarded candidate incremental event {i+1} because required'
                ' fields were missing.'
            )

        except Exception as e:
          logger.warning(
              f'Failed to normalize candidate incremental event {i+1}: {str(e)}'
          )
          continue

      logger.info(
          f'Incremental discovery produced {len(processed_events)} valid'
          ' events.'
      )
      return processed_events

    except json.JSONDecodeError as e:
      logger.error(f'Failed to parse incremental discovery JSON: {str(e)}')
      logger.debug(
          'Incremental discovery raw response preview:'
          f' {full_response[:1000]}...'
      )
      return []

  except ProviderSemanticError:
    raise
  except Exception as e:
    if is_provider_timeout_error(str(e)):
      raise PendingRetryProviderError(
          'gemini_search', 'gemini_search_timeout', str(e)
      ) from e
    if is_llm_rate_limit_error(str(e)):
      raise e
    logger.error(f'Incremental event discovery failed: {str(e)}')
    return []


async def llm_discovery_with_coverage_supervision(
    keyword,
    time_range,
    domain,
    geography_str,
    config,
    significant_peaks,
):
  """Run coverage-supervised LLM discovery across time chunks.

  Args:
      significant_peaks: Significant peaks used to measure event coverage.

  Returns:
      A consolidated list of discovered events.
  """
  logger.info('Running coverage-supervised LLM discovery')

  time_chunks = generate_time_chunks(time_range)
  logger.info(f'Generated {len(time_chunks)} time chunks for discovery')

  all_events = []
  coverage_stats = []

  for i, chunk in enumerate(time_chunks, 1):
    logger.info(
        f"Processing chunk {i}/{len(time_chunks)}: {chunk['start']} to"
        f" {chunk['end']}"
    )

    chunk_peaks = filter_peaks_by_timerange(significant_peaks, chunk)
    logger.info(f'Chunk {i} peak count: {len(chunk_peaks)}')

    chunk_events = await get_initial_events_with_iterations(
        keyword, chunk, domain, geography_str, config, chunk_peaks
    )

    logger.info(f'Chunk {i} discovered event count: {len(chunk_events)}')

    if chunk_peaks:
      chunk_coverage = calculate_coverage_rate(
          chunk_events, chunk_peaks, config
      )
      covered_peaks = len([
          p
          for p in chunk_peaks
          if any(
              abs((p['peak_date'] - ed).days)
              <= config.get('event_sourcing', {}).get(
                  'coverage_time_window_days', 5
              )
              for event in chunk_events
              for ed in extract_event_dates(event)
          )
      ])
    else:
      chunk_coverage = 1.0 if chunk_events else 0.0
      covered_peaks = 0

    coverage_stats.append({
        'time_chunk': i,
        'start_date': chunk['start'],
        'end_date': chunk['end'],
        'total_peaks': len(chunk_peaks),
        'covered_peaks': covered_peaks,
        'uncovered_peaks': len(chunk_peaks) - covered_peaks,
        'total_events': len(chunk_events),
        'coverage_rate': chunk_coverage,
        'keyword': keyword,
    })

    all_events.extend(chunk_events)

  save_coverage_stats(coverage_stats, config)

  logger.info(f'Current event count: {len(all_events)}')
  return all_events


def filter_peaks_by_timerange(
    peaks, time_range
):
  """Keep only peaks that fall inside the provided time range.

  Args:
      peaks: Peak objects with `peak_date`.
      time_range:  {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}

  Returns:
      Peaks that are inside the requested date range.
  """
  if not peaks:
    return []

  start_date = pd.to_datetime(time_range['start'])
  end_date = pd.to_datetime(time_range['end'])

  filtered_peaks = []
  for peak in peaks:
    peak_date = peak['peak_date']
    if start_date <= peak_date <= end_date:
      filtered_peaks.append(peak)

  return filtered_peaks


def calculate_coverage_rate(
    events, peaks, config
):
  """Compute the fraction of peaks covered by discovered events.

  Args:
      events: Discovered events.
      peaks: Significant peaks.
      config: Pipeline configuration.

  Returns:
      Coverage rate in the range [0.0, 1.0].
  """
  if not peaks:
    return 1.0

  if not events:
    return 0.0

  time_window_days = config.get('event_sourcing', {}).get(
      'coverage_time_window_days', 5
  )

  covered_count = 0
  for peak in peaks:
    peak_date = peak['peak_date']

    for event in events:
      event_dates = extract_event_dates(event)

      for event_date in event_dates:
        if abs((peak_date - event_date).days) <= time_window_days:
          covered_count += 1
          break
      else:
        continue
      break

  return covered_count / len(peaks)


def extract_event_dates(event):
  """Extract comparable event dates for coverage matching."""
  dates = []
  date_fields = ['announcement_date', 'occurrence_date']

  for field in date_fields:
    date_str = event.get(field)
    if date_str and date_str != 'null':
      try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        dates.append(date_obj)
      except (ValueError, TypeError):
        continue

  return dates


def save_coverage_stats(coverage_stats, config):
  """CSV

  Args:
      coverage_stats:
      config:
  """
  if not coverage_stats:
    return

  try:

    final_dataset_path = config['paths']['final_dataset']
    output_dir = os.path.dirname(final_dataset_path)
    coverage_file_path = os.path.join(output_dir, 'final_trendcover.csv')

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(coverage_stats)

    df['timestamp'] = datetime.now().isoformat()

    column_order = [
        'timestamp',
        'keyword',
        'time_chunk',
        'start_date',
        'end_date',
        'total_peaks',
        'covered_peaks',
        'uncovered_peaks',
        'total_events',
        'coverage_rate',
    ]
    df = df[column_order]

    file_exists = os.path.exists(coverage_file_path)
    df.to_csv(
        coverage_file_path,
        mode='a',
        header=not file_exists,
        index=False,
        encoding='utf-8',
    )

    logger.info(f'Coverage stats saved to: {coverage_file_path}')

    total_peaks = df['total_peaks'].sum()
    total_covered = df['covered_peaks'].sum()
    overall_coverage = total_covered / total_peaks if total_peaks > 0 else 0.0

    logger.info(
        f'Coverage stats totals: peaks={total_peaks}, covered={total_covered},'
        f' coverage={overall_coverage:.1%}'
    )

  except Exception as e:
    logger.error(f'Failed to save coverage stats: {str(e)}')


def generate_time_chunks(time_range):
  """Split a time range into 3-month blocks.

  Args:
      time_range: Dictionary with 'start' and 'end' keys.

  Returns:
      A list of dictionaries containing 'start' and 'end'.
  """
  start_date = pd.to_datetime(time_range['start'])
  end_date = pd.to_datetime(time_range['end'])

  time_chunks = []
  current_start = start_date

  while current_start <= end_date:
    current_end = min(
        current_start + pd.DateOffset(months=3) - pd.Timedelta(days=1), end_date
    )

    chunk_time_range = {
        'start': current_start.strftime('%Y-%m-%d'),
        'end': current_end.strftime('%Y-%m-%d'),
    }
    time_chunks.append(chunk_time_range)
    current_start = current_end + pd.Timedelta(days=1)

  return time_chunks
