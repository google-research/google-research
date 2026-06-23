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

"""Retry helpers for LLM API rate limiting."""

import functools
import logging
import os
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ProviderSemanticError(RuntimeError):

  def __init__(
      self, terminal_status, reason, provider, detail = ""
  ):
    self.terminal_status = terminal_status
    self.reason = reason
    self.provider = provider
    self.detail = detail
    message = f"{reason} provider={provider}"
    if detail:
      message = f"{message} detail={detail}"
    super().__init__(message)


class WaitingNextDayQuotaError(ProviderSemanticError):

  def __init__(self, provider, detail = ""):
    super().__init__(
        terminal_status="waiting_next_day_quota",
        reason="provider_daily_quota_exhausted",
        provider=provider,
        detail=detail,
    )


class PendingRetryProviderError(ProviderSemanticError):

  def __init__(self, provider, reason, detail = ""):
    super().__init__(
        terminal_status="pending_retry",
        reason=reason,
        provider=provider,
        detail=detail,
    )


class ProviderConfigurationError(ProviderSemanticError):

  def __init__(self, provider, detail = ""):
    super().__init__(
        terminal_status="failed",
        reason="provider_configuration_error",
        provider=provider,
        detail=detail,
    )


def is_llm_rate_limit_error(error_message):
  """Return True when an exception looks like an LLM rate-limit error."""
  rate_limit_indicators = [
      "429",
      "Too Many Requests",
      "rate limit",
      "quota exceeded",
      "RESOURCE_EXHAUSTED",
      "Quota exceeded",
  ]
  error_str = str(error_message).lower()
  return any(
      indicator.lower() in error_str for indicator in rate_limit_indicators
  )


def is_provider_timeout_error(error_message):
  timeout_indicators = [
      "timeout",
      "timed out",
      "deadline exceeded",
      "deadline_exceeded",
      "readtimeout",
      "connecttimeout",
  ]
  error_str = str(error_message).lower()
  return any(indicator in error_str for indicator in timeout_indicators)


def llm_retry_on_429(max_retries = 5):
  """Retry an LLM call on 429-like failures.

  - Maximum retries: 5
  - Exponential backoff starting at 1 second
  - Random jitter of plus or minus 20 percent
  """

  def decorator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
      last_exception = None

      for attempt in range(max_retries + 1):
        try:
          return await func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          if attempt == max_retries:
            if is_llm_rate_limit_error(str(e)):
              logger.error(
                  f"LLM quota exhausted after {max_retries} retries: {str(e)}"
              )
              raise WaitingNextDayQuotaError("gemini", str(e)) from e
            logger.error(
                f"LLM call failed after {max_retries} retries: {str(e)}"
            )
            raise e

          if is_llm_rate_limit_error(str(e)):
            base_delay = min(3**attempt, 1800)
            jitter = random.uniform(-0.2, 0.2) * base_delay
            delay = max(1, base_delay + jitter)

            logger.warning(
                f"LLM rate limit detected on attempt {attempt + 1}. Retrying in"
                f" {delay:.1f}s: {str(e)}"
            )
            time.sleep(delay)
            continue
          else:
            logger.error(
                f"LLM call failed with a non-retryable error: {str(e)}"
            )
            raise e

      raise last_exception

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
      last_exception = None

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          if attempt == max_retries:
            if is_llm_rate_limit_error(str(e)):
              logger.error(
                  f"LLM quota exhausted after {max_retries} retries: {str(e)}"
              )
              raise WaitingNextDayQuotaError("gemini", str(e)) from e
            logger.error(
                f"LLM call failed after {max_retries} retries: {str(e)}"
            )
            raise e

          if is_llm_rate_limit_error(str(e)):
            base_delay = min(3**attempt, 1800)
            jitter = random.uniform(-0.2, 0.2) * base_delay
            delay = max(1, base_delay + jitter)

            logger.warning(
                f"LLM rate limit detected on attempt {attempt + 1}. Retrying in"
                f" {delay:.1f}s: {str(e)}"
            )
            time.sleep(delay)
            continue
          else:
            logger.error(
                f"LLM call failed with a non-retryable error: {str(e)}"
            )
            raise e

      raise last_exception

    return (
        async_wrapper
        if getattr(func, "__code__", None) and func.__code__.co_flags & 0x80
        else sync_wrapper
    )

  return decorator


def llm_retry_on_429_20flash(max_retries = 5):
  """Retry a Gemini 2.0 Flash call on 429-like failures.

  - Maximum retries: 5
  - Exponential backoff starting at 1 second
  - Random jitter of plus or minus 20 percent
  """

  def decorator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
      last_exception = None

      for attempt in range(max_retries + 1):
        try:
          return await func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          if attempt == max_retries:
            if is_llm_rate_limit_error(str(e)):
              logger.error(
                  f"LLM quota exhausted after {max_retries} retries: {str(e)}"
              )
              raise WaitingNextDayQuotaError("gemini", str(e)) from e
            logger.error(
                f"LLM call failed after {max_retries} retries: {str(e)}"
            )
            raise e

          if is_llm_rate_limit_error(str(e)):
            base_delay = min(3**attempt, 180)
            jitter = random.uniform(-0.2, 0.2) * base_delay
            delay = max(1, base_delay + jitter)

            logger.warning(
                f"LLM rate limit detected on attempt {attempt + 1}. Retrying in"
                f" {delay:.1f}s: {str(e)}"
            )
            time.sleep(delay)
            continue
          else:
            logger.error(
                f"LLM call failed with a non-retryable error: {str(e)}"
            )
            raise e

      raise last_exception

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
      last_exception = None

      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_exception = e

          if attempt == max_retries:
            if is_llm_rate_limit_error(str(e)):
              logger.error(
                  f"LLM quota exhausted after {max_retries} retries: {str(e)}"
              )
              raise WaitingNextDayQuotaError("gemini", str(e)) from e
            logger.error(
                f"LLM call failed after {max_retries} retries: {str(e)}"
            )
            raise e

          if is_llm_rate_limit_error(str(e)):
            base_delay = min(2**attempt, 180)
            jitter = random.uniform(-0.2, 0.2) * base_delay
            delay = max(1, base_delay + jitter)

            logger.warning(
                "Gemini 2.0 Flash rate limit detected on attempt"
                f" {attempt + 1}. Retrying in {delay:.1f}s: {str(e)}"
            )
            time.sleep(delay)
            continue
          else:
            logger.error(
                "Gemini 2.0 Flash call failed with a non-retryable error:"
                f" {str(e)}"
            )
            raise e

      raise last_exception

    return (
        async_wrapper
        if getattr(func, "__code__", None) and func.__code__.co_flags & 0x80
        else sync_wrapper
    )

  return decorator
