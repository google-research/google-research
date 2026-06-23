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

"""LLM Handler for Gemini API interactions"""

import json
import logging
import os
from typing import List, Optional

from google import genai
from google.genai import types

from .llm_retry_handler import (
    PendingRetryProviderError,
    ProviderSemanticError,
    is_llm_rate_limit_error,
    is_provider_timeout_error,
    llm_retry_on_429,
    llm_retry_on_429_20flash,
)
from .utils import clean_llm_json_output

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RECALIBRATION_SEARCH_TIMEOUT_MS = int(
    os.getenv("GEMINI_RECALIBRATION_HTTP_TIMEOUT_MS", "60000")
)


def _create_genai_client(
    api_key, timeout_ms = None
):
  original_google_api_key = os.environ.pop("GOOGLE_API_KEY", None)
  try:
    if timeout_ms and timeout_ms > 0:
      http_options = types.HttpOptions(timeout=timeout_ms)
      return genai.Client(api_key=api_key, http_options=http_options)
    return genai.Client(api_key=api_key)
  finally:
    if original_google_api_key is not None:
      os.environ["GOOGLE_API_KEY"] = original_google_api_key


@llm_retry_on_429_20flash(max_retries=3)
async def get_single_source_summary(
    content,
    source_url,
    source_query,
    keywords,
    config,
):
  """Analyzes a single webpage content to extract a structured event summary.

  Args:
      content: The content of the webpage.
      source_url: The source URL of the content.
      source_query: The original query that led to discovering this content.
      keywords: The search keywords that led to this event.
      config: The application configuration.

  Returns:
      A dictionary with the structured summary, or None if an error occurs.
  """
  logger.info(f"Analyzing content from source: {source_url}")
  try:
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
      logger.error("GEMINI_API_KEY not found in environment.")
      return None

    client = _create_genai_client(api_key)

    # Get model from config, with a default
    model_name = config.get("llm_models", {}).get(
        "single_source_summary", "gemini-3.5-flash"
    )
    logger.info(
        "single_source_summary:"
        f" {config.get('llm_models', {}).get('single_source_summary', 'not configured')}"
    )
    logger.info(f"Using model for single-source summary: {model_name}")

    keywords_str = ", ".join(keywords)

    prompt = f"""
        CRITICAL INSTRUCTIONS:
        1.  Your primary directive is to act as a factual data extractor. You are forbidden from inferring, guessing, or using any information not explicitly present in the "Web Page Content" provided below.
        2.  Your entire response MUST be based *only* on the provided text. DO NOT HALLUCINATE. If you invent any information, the task is considered a failure.
        3.  If the text does not contain a specific, factual event relevant to the keywords, you MUST set the "event_summary" to "Not relevant" and other fields to null.

        For context, this content was discovered while researching the query: '{source_query}'. Please analyze the text and provide a summary that addresses this research context.

        As an information extraction engine, your task is to analyze the following web page content for events relevant to the keywords: {keywords_str}.

        Web Page Content:
        \"\"\"
        {content[:1000000]}
        \"\"\"

        Your response MUST be a single JSON object with the exact following structure. Do not add any commentary before or after the JSON.
        {{
          "event_summary": "A concise, one-sentence summary of the core factual event. Must be directly supported by the text.",
          "event_date": "YYYY-MM-DD format. Set to null if not found in the text.",
          "location": "The specific location of the event. Set to null if not found in the text.",
          "supporting_quote": "An EXACT quote from the web page content that directly and explicitly supports the event_summary. Set to null if no relevant event is found.",
          "source_url": "{source_url}"
        }}
        """

    response = await client.aio.models.generate_content(
        model=model_name, contents=prompt
    )
    cleaned_response = clean_llm_json_output(response.text)

    summary_data = json.loads(cleaned_response)
    logger.info(f"Successfully extracted summary from {source_url}")
    return summary_data

  except Exception as e:
    if is_llm_rate_limit_error(str(e)):
      raise e
    logger.error(f"Error analyzing single source {source_url}: {e}")
    return None


@llm_retry_on_429(max_retries=5)
async def verify_claim_with_evidence(
    event_claim, source_evidence, config
):
  """LLM

  Apply a suppression radius to avoid reporting overlapping peaks.

  Args:
      event_claim:
      source_evidence: URL
      config:

  Returns:
      verified_statements
  """
  try:

    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
      logger.error("GEMINI_API_KEY not found")
      return {
          "page_status": "provider_configuration_error",
          "verified_statements": [],
          "overall_timing": {"content_date": None, "publish_date": None},
          "reasoning": "API key not available",
      }

    client = _create_genai_client(api_key)

    model_name = config.get("llm_models", {}).get(
        "claim_verifier", "gemini-3.5-flash"
    )
    logger.info(
        "claim_verifier:"
        f" {config.get('llm_models', {}).get('claim_verifier', 'not configured')}"
    )
    logger.info(f"Using model for claim verification: {model_name}")

    prompt = f"""You are a meticulous fact-checker. Your task is to verify claims against source content and identify invalid pages.

**CRITICAL: First determine if the source content is valid.**

**Important Definitions:**
- **content_date**: The date when the event itself occurred (e.g., product launch, announcement)
- **publish_date**: The original publication date of the source (NOT "updated" or "last modified" dates)

**Few-Shot Learning Examples:**

**Example 1 - Valid Content:**
Event Claim: "Apple Vision Pro will be available on February 2, 2024"
Source Text:
```
<h2>Apple Vision Pro Available in the U.S. on February 2</h2>
<span class="publish-date">POSTED ON JANUARY 8, 2024</span>
<p>Apple today announced Apple Vision Pro will be available beginning Friday, February 2...</p>
<footer>Last updated: January 10, 2024</footer>
```
Expected Output:
```json
{{
  "page_status": "valid_content",
  "verified_statements": [
    {{
      "statement": "Apple Vision Pro will be available on February 2, 2024",
      "status": "Confirmed",
      "supporting_quote": "Apple Vision Pro will be available beginning Friday, February 2"
    }}
  ],
  "overall_timing": {{
    "content_date": "2024-02-02",
    "publish_date": "2024-01-08"
  }},
  "reasoning": "Valid press release content. Used original publish date (Jan 8), ignored 'last updated' footer."
}}
```

**Example 2 - 404 Error Page:**
Event Claim: "iPhone 16 rumors surface in March 2024"
Source Text:
```
<title>Page Not Found - TechNews</title>
<h1>404 - Page Not Found</h1>
<p>The page you're looking for doesn't exist.</p>
<div class="sidebar">Today's Hot Topics: July 28, 2025</div>
```
Expected Output:
```json
{{
  "page_status": "error_page_404",
  "verified_statements": [],
  "overall_timing": {{
    "content_date": null,
    "publish_date": null
  }},
  "reasoning": "This is a 404 error page with no valid content. Sidebar dates are irrelevant template content."
}}
```

**Example 3 - Access Denied:**
Event Claim: "Samsung Galaxy S25 specs leaked"
Source Text:
```
<h1>Access Denied</h1>
<p>You don't have permission to view this page. Please log in.</p>
<div class="trending">Trending now: March 15, 2025</div>
```
Expected Output:
```json
{{
  "page_status": "access_denied",
  "verified_statements": [],
  "overall_timing": {{
    "content_date": null,
    "publish_date": null
  }},
  "reasoning": "Access denied page. No verifiable content available."
}}
```

**Now analyze the actual content:**

**1. The Event Claim to Analyze:**
{event_claim}

**2. The Evidence (Source Text):**
---
{source_evidence}
---

**Instructions:**
1. First, determine page_status: valid_content, error_page_404, access_denied, or login_wall
2. If page_status is NOT "valid_content", return empty verified_statements and null dates
3. If valid_content, decompose claim into atomic facts and verify each one
4. For dates: Use ORIGINAL publish dates, ignore "updated", "modified", or sidebar dates
5. Classify fact status: Confirmed, Anticipated, Speculation, or Not_Found

**JSON Output format:**
{{
  "page_status": "<valid_content|error_page_404|access_denied|login_wall>",
  "verified_statements": [
    {{
      "statement": "<atomic factual statement>",
      "status": "<Confirmed|Anticipated|Speculation|Not_Found>",
      "supporting_quote": "<exact quote from text or null>"
    }}
  ],
  "overall_timing": {{
    "content_date": "<YYYY-MM-DD when the event occurred or null>",
    "publish_date": "<YYYY-MM-DD when source was originally published or null>"
  }},
  "reasoning": "<Brief explanation of your analysis process>"
}}

Respond with ONLY the JSON object, no additional text."""

    response = client.models.generate_content(model=model_name, contents=prompt)
    response_text = response.text.strip()

    logger.debug("Received raw claim verification response from the LLM.")

    try:

      if response_text.startswith("```json"):
        response_text = response_text[7:]
      if response_text.endswith("```"):
        response_text = response_text[:-3]
      response_text = response_text.strip()

      result = json.loads(response_text)

      if "page_status" not in result:
        result["page_status"] = "invalid_verifier_response"
      if "verified_statements" not in result:
        result["verified_statements"] = []
      if "overall_timing" not in result:
        result["overall_timing"] = {"content_date": None, "publish_date": None}
      if "reasoning" not in result:
        result["reasoning"] = "No reasoning provided"

      logger.debug(
          "Verification diagnostics: Final parsed dictionary being returned:"
          f" {result}"
      )

      return result

    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse JSON response: {e}")
      logger.warning(f"Response text: {response_text}")
      return {
          "page_status": "verification_failed",
          "verified_statements": [],
          "overall_timing": {"content_date": None, "publish_date": None},
          "reasoning": f"Failed to parse LLM response: {str(e)}",
      }

  except Exception as e:
    if is_llm_rate_limit_error(str(e)):
      raise e
    logger.error(f"Error in claim verification: {str(e)}")
    return {
        "page_status": "verification_failed",
        "verified_statements": [],
        "overall_timing": {"content_date": None, "publish_date": None},
        "reasoning": f"Verification failed: {str(e)}",
    }


@llm_retry_on_429(max_retries=5)
async def recalibrate_event_claim_with_search(
    original_claim, start_date, end_date, config
):
  """Use search-grounded LLM to recalibrate and re-verify an event claim

  Args:
      original_claim: The original event claim that couldn't be verified
      start_date: Start date of the time window (YYYY-MM-DD)
      end_date: End date of the time window (YYYY-MM-DD)
      config: Configuration dictionary

  Returns:
      Dictionary with recalibration_status and event object
  """
  logger.info(f"Starting claim recalibration for: {original_claim[:100]}...")

  try:
    # API setup
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
      logger.error("GEMINI_API_KEY not found")
      return {"recalibration_status": "unverified", "event": None}

    client = _create_genai_client(
        api_key, timeout_ms=RECALIBRATION_SEARCH_TIMEOUT_MS
    )

    # Get model from config
    model_name = (
        config.get("event_sourcing", {})
        .get("recalibration", {})
        .get("recalibration_model", "gemini-3.5-flash")
    )
    logger.debug(f"Using model for claim recalibration: {model_name}")

    prompt = f"""
You are an expert fact-checker and research analyst.

Your task is to re-investigate and verify the following event claim using your web search capabilities:

**Original Event Claim to Verify:**
{original_claim}

**Time Window:** {start_date} to {end_date}

**Your Mission:**
1. Use your web search to find reliable sources about this event
2. Verify if the event actually occurred and when
3. Extract accurate date information
4. Provide corrected/verified event details with supporting URLs

**Date Extraction Guidelines:**
- **announcement_date**: When the news was first published/announced
- **occurrence_date**: When the event actually happened or was scheduled
- If dates cannot be verified, use `null`

**Output Requirements:**
Respond with ONLY a JSON object in this exact format:

```json
{{
  "recalibration_status": "recalibrated_and_verified" or "unverified",
  "event": {{
    "event_summary": "Verified and corrected event description",
    "announcement_date": "YYYY-MM-DD or null",
    "occurrence_date": "YYYY-MM-DD or null",
    "event_type": "Instantaneous|Anticipated|Retrospective|Protracted or null",
    "source_urls": ["url1", "url2"],
    "confidence_score": 0.8
  }}
}}
```

If the event cannot be verified or no reliable sources are found, set "recalibration_status" to "unverified" and "event" to null.
"""

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    generate_content_config = types.GenerateContentConfig(
        max_output_tokens=65000, tools=tools
    )

    logger.info("Sending structured JSON extraction request to the LLM.")
    response_chunks = []

    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    ):
      if chunk.text:
        response_chunks.append(chunk.text)

    full_response = "".join(response_chunks)
    logger.info(
        f"Received structured LLM response with {len(full_response)}"
        " characters."
    )

    try:

      json_start = full_response.find("{")
      json_end = full_response.rfind("}") + 1

      if json_start == -1 or json_end == 0:
        logger.warning(
            "No JSON object was found in the claim recalibration response."
        )
        return {"recalibration_status": "unverified", "event": None}

      json_text = full_response[json_start:json_end]
      result_data = json.loads(json_text)

      logger.info(
          "Claim recalibration status:"
          f" {result_data.get('recalibration_status', 'unknown_status')}"
      )
      return result_data

    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse claim recalibration JSON: {str(e)}")
      logger.debug(
          f"Claim recalibration raw response preview: {full_response[:1000]}..."
      )
      return {"recalibration_status": "unverified", "event": None}

  except ProviderSemanticError:
    raise
  except Exception as e:
    if is_provider_timeout_error(str(e)):
      raise PendingRetryProviderError(
          "gemini_search", "gemini_search_timeout", str(e)
      ) from e
    if is_llm_rate_limit_error(str(e)):
      raise e
    logger.error(f"Claim recalibration failed: {str(e)}")
    return {"recalibration_status": "unverified", "event": None}
