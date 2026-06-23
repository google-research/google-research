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

import logging
import re

logger = logging.getLogger(__name__)


def clean_llm_json_output(text):
  """Cleans the raw string output from an LLM to extract a valid JSON object or array.

  It strips markdown code blocks and other surrounding text.

  Args:
      text: The raw string output from the LLM.

  Returns:
      A cleaned string that should be a valid JSON.
  """
  logger.debug(f'Cleaning raw LLM output: {text}')

  # Remove markdown code block fences (e.g., ```json ... ```)
  cleaned_text = re.sub(r'```json\s*', '', text, flags=re.MULTILINE)
  cleaned_text = re.sub(r'```', '', cleaned_text)
  cleaned_text = cleaned_text.strip()

  # Find the start and end of the main JSON object or array
  json_start = -1
  json_end = -1

  first_brace = cleaned_text.find('{')
  first_bracket = cleaned_text.find('[')

  if first_brace == -1 and first_bracket == -1:
    logger.warning('No JSON object or array found in the LLM output.')
    return cleaned_text  # Return as-is if no JSON structure is apparent

  if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
    # It's likely a JSON object
    json_start = first_brace
    json_end = cleaned_text.rfind('}')
  else:
    # It's likely a JSON array
    json_start = first_bracket
    json_end = cleaned_text.rfind(']')

  if json_start != -1 and json_end != -1 and json_end > json_start:
    final_json = cleaned_text[json_start : json_end + 1]
    logger.debug(f'Extracted JSON string: {final_json}')
    return final_json

  logger.warning(
      'Could not extract a valid JSON structure from the LLM output.'
  )
  return cleaned_text  # Fallback
