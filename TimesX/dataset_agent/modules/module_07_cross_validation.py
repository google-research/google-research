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

"""Module 07: Cross Validation and Task Allocation"""

from datetime import datetime, timedelta
import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def cross_validate_and_allocate(
    initial_events, significant_peaks, config
):
  """Cross-validate LLM-discovered events against significant trend peaks and

  allocate validation and exploration tasks.

  Args:
      initial_events: Events proposed by the LLM discovery stage.
      significant_peaks: Significant peaks detected from Google Trends.
      config: Pipeline configuration.

  Returns:
      A tuple of `(validation_tasks, exploration_tasks)`.
  """
  logger.info(
      "Cross-validating %s initial events against %s significant peaks",
      len(initial_events),
      len(significant_peaks),
  )

  cross_validation_config = config.get("cross_validation", {})
  time_window_days = cross_validation_config.get("time_window_matching_days", 5)
  confidence_threshold = cross_validation_config.get(
      "confidence_threshold", 0.7
  )

  logger.info(
      "Cross-validation settings: time_window_days=%s, confidence_threshold=%s",
      time_window_days,
      confidence_threshold,
  )

  covered_peaks = set()
  validation_tasks = []

  for event in initial_events:
    if event.get("confidence_score", 0) < confidence_threshold:
      logger.info(
          "Skipping low-confidence event: %s...",
          event.get("event_summary", "")[:50],
      )
      continue

    event_dates = extract_event_dates(event)
    matched_peaks = find_matching_peaks(
        event_dates, significant_peaks, time_window_days
    )

    validation_task = create_validation_task(event, matched_peaks, config)
    validation_tasks.append(validation_task)

    for peak in matched_peaks:
      covered_peaks.add(peak["peak_date"])

    logger.info(
        "Allocated validation task for event '%s...' with %s matched peaks",
        event.get("event_summary", "")[:50],
        len(matched_peaks),
    )

  logger.info("Covered peaks count: %s", len(covered_peaks))

  uncovered_peaks = []
  for peak in significant_peaks:
    if peak["peak_date"] not in covered_peaks:
      uncovered_peaks.append(peak)

  logger.info("Uncovered peaks count: %s", len(uncovered_peaks))

  exploration_tasks = []
  for peak in uncovered_peaks:
    exploration_task = create_exploration_task(peak, config)
    exploration_tasks.append(exploration_task)

    logger.info(
        "Allocated exploration task for peak %s (peak_value=%.1f)",
        peak["peak_date"].date(),
        peak["peak_value"],
    )

  validation_tasks = assign_task_priorities(validation_tasks, "validation")
  exploration_tasks = assign_task_priorities(exploration_tasks, "exploration")

  logger.info(
      "Task allocation complete: %s validation tasks, %s exploration tasks",
      len(validation_tasks),
      len(exploration_tasks),
  )

  return validation_tasks, exploration_tasks


def extract_event_dates(event):
  """Extract event dates used for cross-validation matching.

  Args:
      event: Event dictionary.

  Returns:
      Parsed event dates.
  """
  dates = []
  date_fields = ["announcement_date", "occurrence_date"]

  for field in date_fields:
    date_str = event.get(field)
    if date_str and date_str != "null":
      try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        dates.append(date_obj)
      except (ValueError, TypeError):
        logger.warning("Skipping invalid event date: %s", date_str)
        continue

  return dates


def find_matching_peaks(
    event_dates,
    significant_peaks,
    time_window_days,
):
  """Find significant peaks that fall within the matching window of an event.

  Args:
      event_dates: Parsed event dates.
      significant_peaks: Significant trend peaks.
      time_window_days: Allowed matching window in days.

  Returns:
      Peaks matched to the event.
  """
  matched_peaks = []

  for peak in significant_peaks:
    peak_date = peak["peak_date"]

    for event_date in event_dates:
      days_diff = abs((peak_date - event_date).days)
      if days_diff <= time_window_days:
        matched_peaks.append(peak)
        break

  return matched_peaks


def create_validation_task(
    event, matched_peaks, config
):
  """Create a validation task for a discovered event.

  Args:
      event: Event to validate.
      matched_peaks: Peaks matched to the event.
      config: Pipeline configuration.

  Returns:
      Validation task payload.
  """
  task_config = config.get("task_execution", {}).get("validation_tasks", {})

  validation_task = {
      "task_id": f"validation_{event['event_id']}",
      "task_type": "validation",
      "event_data": event,
      "matched_peaks": matched_peaks,
      "query_budget": task_config.get("query_budget_per_event", 3),
      "crawl_count_per_query": task_config.get("crawl_count_per_query", 5),
      "priority": calculate_validation_priority(event, matched_peaks),
      "status": "pending",
      "created_at": datetime.now().isoformat(),
  }

  return validation_task


def create_exploration_task(peak, config):
  """Create an exploration task for an uncovered peak.

  Args:
      peak: Peak to investigate.
      config: Pipeline configuration.

  Returns:
      Exploration task payload.
  """
  task_config = config.get("task_execution", {}).get("exploration_tasks", {})

  exploration_task = {
      "task_id": f"exploration_{peak['peak_date'].strftime('%Y%m%d')}",
      "task_type": "exploration",
      "peak_data": peak,
      "query_budget": task_config.get("query_budget_per_peak", 5),
      "crawl_count_per_query": task_config.get("crawl_count_per_query", 8),
      "max_time_minutes": task_config.get("max_exploration_time_minutes", 10),
      "priority": calculate_exploration_priority(peak),
      "status": "pending",
      "created_at": datetime.now().isoformat(),
  }

  return exploration_task


def calculate_validation_priority(
    event, matched_peaks
):
  """Calculate validation priority for an event task.

  Args:
      event: Event dictionary.
      matched_peaks: Peaks matched to the event.

  Returns:
      Priority score in the range [0.0, 1.0].
  """
  priority = 0.5

  confidence = event.get("confidence_score", 0.5)
  priority += confidence * 0.3

  if matched_peaks:
    peak_bonus = min(len(matched_peaks) * 0.1, 0.2)
    priority += peak_bonus

    max_peak_value = max([p["peak_value"] for p in matched_peaks])
    if max_peak_value > 80:
      priority += 0.1

  return min(priority, 1.0)


def calculate_exploration_priority(peak):
  """Calculate exploration priority for an uncovered peak.

  Args:
      peak: Peak dictionary.

  Returns:
      Priority score in the range [0.0, 1.0].
  """
  priority = 0.3

  peak_value = peak.get("peak_value", 0)
  if peak_value > 90:
    priority += 0.3
  elif peak_value > 80:
    priority += 0.2
  elif peak_value > 70:
    priority += 0.1

  percentile_rank = peak.get("percentile_rank", 0)
  priority += percentile_rank * 0.2

  if peak.get("detection_method") == "absolute":
    priority += 0.1

  return min(priority, 1.0)


def assign_task_priorities(tasks, task_type):
  """Sort tasks by priority and assign execution order.

  Args:
      tasks: Task payloads.
      task_type: Task category name for logging.

  Returns:
      Sorted tasks with execution order assigned.
  """
  if not tasks:
    return tasks

  sorted_tasks = sorted(tasks, key=lambda x: x["priority"], reverse=True)

  for i, task in enumerate(sorted_tasks):
    task["execution_order"] = i + 1

  logger.info(
      "%s task priority assignment complete: %s tasks",
      task_type,
      len(sorted_tasks),
  )

  return sorted_tasks


def generate_task_summary(
    validation_tasks, exploration_tasks
):
  """Summarize the allocated validation and exploration tasks.

  Args:
      validation_tasks: Validation task list.
      exploration_tasks: Exploration task list.

  Returns:
      Summary statistics for allocated tasks.
  """
  summary = {
      "total_tasks": len(validation_tasks) + len(exploration_tasks),
      "validation_tasks": {
          "count": len(validation_tasks),
          "total_query_budget": sum(
              t["query_budget"] for t in validation_tasks
          ),
          "high_priority_count": len(
              [t for t in validation_tasks if t["priority"] > 0.7]
          ),
      },
      "exploration_tasks": {
          "count": len(exploration_tasks),
          "total_query_budget": sum(
              t["query_budget"] for t in exploration_tasks
          ),
          "high_priority_count": len(
              [t for t in exploration_tasks if t["priority"] > 0.7]
          ),
      },
      "total_query_budget": sum(
          t["query_budget"] for t in validation_tasks + exploration_tasks
      ),
  }

  return summary
