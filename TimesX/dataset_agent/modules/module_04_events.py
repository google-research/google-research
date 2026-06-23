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

"""Module 04: Event Detection and Annotation"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Dict, List, Tuple
import uuid
from crawl4ai import AsyncWebCrawler
from google import genai
from google.genai import types
import requests
from .evidence_cleaning import clean_crawl_evidence, cleaning_metadata
from .llm_handler import get_single_source_summary, recalibrate_event_claim_with_search, verify_claim_with_evidence
from .llm_retry_handler import (
    PendingRetryProviderError,
    ProviderConfigurationError,
    ProviderSemanticError,
    WaitingNextDayQuotaError,
    is_llm_rate_limit_error,
    llm_retry_on_429,
)

logger = logging.getLogger(__name__)


def _classify_customsearch_error(response):
  try:
    payload = response.json()
  except Exception:
    payload = {}
  errors = (
      ((payload.get("error") or {}).get("errors") or [])
      if isinstance(payload, dict)
      else []
  )
  reasons = [
      str(item.get("reason", "")).strip()
      for item in errors
      if isinstance(item, dict)
  ]
  reason_text = " ".join(reasons).lower()
  body_text = response.text.lower() if getattr(response, "text", None) else ""
  if "dailylimitexceeded" in reason_text or "daily limit exceeded" in body_text:
    return "daily"
  if (
      "userratelimitexceeded" in reason_text
      or "ratelimitexceeded" in reason_text
      or response.status_code == 429
      or "quota exceeded" in body_text
  ):
    return "transient"
  return "other"


def _create_gemini_client(api_key):
  original_google_api_key = os.environ.pop("GOOGLE_API_KEY", None)
  try:
    return genai.Client(api_key=api_key)
  finally:
    if original_google_api_key is not None:
      os.environ["GOOGLE_API_KEY"] = original_google_api_key


class ResearchTreeManager:
  """A manager for the research tree structure and state during iterative research."""

  def __init__(self):
    self.nodes = {}  # node_id -> node_data
    self.query_to_id = {}  # query -> node_id mapping
    self.root_id = None
    self.completed_nodes = set()

  def create_node(
      self,
      query,
      node_type = "query",
      parent_id = None,
      priority = 1.0,
  ):
    """Create a new research node and return its ID"""
    node_id = str(uuid.uuid4())
    self.query_to_id[query] = node_id

    node_data = {
        "id": node_id,
        "query": query,
        "node_type": node_type,  # 'root', 'hypothesis', 'query'
        "parent_id": parent_id,
        "children_ids": [],
        "learnings": [],
        "sources": [],
        "status": "in_progress",
        "depth": 0 if parent_id is None else self.nodes[parent_id]["depth"] + 1,
        "priority": priority,
        "creation_time": datetime.now().isoformat(),
        "metadata": {},
    }

    # Add queries_spent field for hypothesis nodes
    if node_type == "hypothesis":
      node_data["queries_spent"] = 0

    self.nodes[node_id] = node_data

    # Update parent's children list
    if parent_id and parent_id in self.nodes:
      self.nodes[parent_id]["children_ids"].append(node_id)

    # Set as root if this is the first node
    if self.root_id is None:
      self.root_id = node_id

    return node_id

  def add_learnings(self, node_id, learnings):
    """Add learnings to a specific node"""
    if node_id in self.nodes:
      self.nodes[node_id]["learnings"].extend(learnings)

  def add_sources(self, node_id, sources):
    """Add sources to a specific node"""
    if node_id in self.nodes:
      self.nodes[node_id]["sources"].extend(sources)

  def add_metadata(self, node_id, metadata):
    """Add or update metadata for a specific node"""
    if node_id in self.nodes:
      self.nodes[node_id].setdefault("metadata", {}).update(metadata)

  def complete_node(self, node_id):
    """Mark a node as completed"""
    if node_id in self.nodes:
      self.nodes[node_id]["status"] = "completed"
      self.completed_nodes.add(node_id)

  def get_all_learnings(self):
    """Get all learnings from all nodes"""
    all_learnings = []
    for node in self.nodes.values():
      all_learnings.extend(node["learnings"])
    return all_learnings

  def update_node_status(self, node_id, new_status):
    """Update the status of a specific node"""
    if node_id in self.nodes:
      self.nodes[node_id]["status"] = new_status
      if new_status == "completed":
        self.completed_nodes.add(node_id)

  def update_node_query(self, node_id, new_query):
    """Update the query text of a specific node"""
    if node_id in self.nodes:
      old_query = self.nodes[node_id]["query"]
      self.nodes[node_id]["query"] = new_query
      # Update the query to ID mapping
      if old_query in self.query_to_id:
        del self.query_to_id[old_query]
      self.query_to_id[new_query] = node_id

  def get_active_hypotheses(self):
    """Get all hypothesis nodes that are currently in progress"""
    active_hypotheses = []
    for node in self.nodes.values():
      if node["node_type"] == "hypothesis" and node["status"] == "in_progress":
        active_hypotheses.append(node)
    return active_hypotheses

  def get_pending_queries_by_priority(self):
    """Get all pending query nodes sorted by priority (highest first)"""
    pending_queries = []
    for node in self.nodes.values():
      if node["node_type"] == "query" and node["status"] == "in_progress":
        pending_queries.append(node)

    # Sort by priority (highest first), then by creation time (oldest first)
    pending_queries.sort(key=lambda x: (-x["priority"], x["creation_time"]))
    return pending_queries

  def update_node_priority(self, node_id, new_priority):
    """Update the priority of a specific node"""
    if node_id in self.nodes:
      self.nodes[node_id]["priority"] = new_priority

  def increment_hypothesis_query_count(self, hypothesis_id):
    """Increment the queries_spent count for a hypothesis"""
    if (
        hypothesis_id in self.nodes
        and self.nodes[hypothesis_id]["node_type"] == "hypothesis"
    ):
      self.nodes[hypothesis_id]["queries_spent"] = (
          self.nodes[hypothesis_id].get("queries_spent", 0) + 1
      )

  def get_hypothesis_query_count(self, hypothesis_id):
    """Get the queries_spent count for a hypothesis"""
    if (
        hypothesis_id in self.nodes
        and self.nodes[hypothesis_id]["node_type"] == "hypothesis"
    ):
      return self.nodes[hypothesis_id].get("queries_spent", 0)
    return 0

  def get_hypothesis_completion_rate(self, hypothesis_id):
    """Calculate the completion rate of a hypothesis based on its child queries"""
    if hypothesis_id not in self.nodes:
      return 0.0

    hypothesis = self.nodes[hypothesis_id]
    if not hypothesis["children_ids"]:
      return 0.0

    completed_children = sum(
        1
        for child_id in hypothesis["children_ids"]
        if self.nodes[child_id]["status"] == "completed"
    )
    return completed_children / len(hypothesis["children_ids"])

  def build_tree_json(self):
    """Build the complete tree structure as JSON"""
    if not self.root_id:
      return {}

    def build_node(node_id):
      node = self.nodes[node_id]
      result = {
          "id": node["id"],
          "query": node["query"],
          "node_type": node["node_type"],
          "parent_id": node["parent_id"],
          "status": node["status"],
          "depth": node["depth"],
          "learnings": node["learnings"],
          "sources": node["sources"],
          "children": [
              build_node(child_id) for child_id in node["children_ids"]
          ],
          "metadata": node.get("metadata", {}),
      }

      # Add debugging information for query nodes
      if node["node_type"] == "query":
        result["priority"] = node.get("priority", 1.0)
        result["creation_time"] = node.get("creation_time", "unknown")

      return result

    return build_node(self.root_id)


class IterativeResearcher:
  """Conducts iterative, deep research using a three-stage approach - REFACTORED for new task types"""

  def __init__(self, config, task_type, task_data):
    """Research flow helper.

    Args:
        config: Pipeline configuration dictionary.
        task_type: Either "validation" or "exploration".
        task_data: Task payload required by the selected task type.
    """
    self.config = config
    self.task_type = task_type
    self.task_data = task_data
    self.tree_manager = ResearchTreeManager()

    if task_type == "validation":
      self.event_data = task_data["event_data"]
      self.keywords = [config["target_variable"]]

      self.source_urls_from_llm = self.event_data.get("source_urls", [])

      announcement_date = self.event_data.get("announcement_date")
      if announcement_date:
        try:
          self.peak_date = datetime.strptime(announcement_date, "%Y-%m-%d")
        except ValueError:
          self.peak_date = datetime.now()
      else:

        self.peak_date = datetime.now()
    elif task_type == "exploration":
      self.peak_data = task_data["peak_data"]
      self.keywords = [config["target_variable"]]
      self.peak_date = self.peak_data["peak_date"]
      self.source_urls_from_llm = []
    else:
      raise ValueError(f"Unsupported task_type: {task_type}")

    task_config = config.get("task_execution", {})
    if task_type == "validation":
      task_specific_config = task_config.get("validation_tasks", {})
    else:
      task_specific_config = task_config.get("exploration_tasks", {})

    self.query_budget = task_data.get(
        "query_budget", task_specific_config.get("query_budget_per_event", 3)
    )
    self.crawl_count_per_query = task_data.get(
        "crawl_count_per_query",
        task_specific_config.get("crawl_count_per_query", 5),
    )

    self.iterative_crawl_count = self.crawl_count_per_query
    self.initial_source_urls_processed = 0
    self.followup_search_queries_executed = 0

  async def investigate_validation_task(self):
    """- URL

    Returns:
        Tuple of (final_summary, research_tree)
    """
    logger.info(f"Starting validation task: {self.task_data['task_id']}")

    event_summary = self.event_data.get("event_summary", "Unknown event")
    root_question = f"Verify and enrich details about: {event_summary}"
    root_id = self.tree_manager.create_node(root_question, node_type="root")

    try:

      logger.info("Processing source URLs returned by the initial LLM step.")
      if self.source_urls_from_llm:
        logger.info(
            f"Initial LLM step returned {len(self.source_urls_from_llm)} source"
            " URLs."
        )
        for i, url in enumerate(self.source_urls_from_llm):
          logger.info(f"URL {i+1}/{len(self.source_urls_from_llm)}: {url}")

          self.initial_source_urls_processed += 1
          node_id = await self._process_url_directly(url, root_id)
      else:
        logger.info("No source URLs were returned from the initial LLM step.")

      logger.info("Stage 2: evaluating information sufficiency.")
      sufficiency_result = await self._evaluate_and_plan_next_steps()

      if not sufficiency_result["is_sufficient"]:
        logger.info(
            "Stage 3: planned next actions count ="
            f" {len(sufficiency_result['next_actions'])}"
        )
        await self._execute_action_plan(
            sufficiency_result["next_actions"], root_id
        )
      else:
        logger.info(
            "Stage 3: evidence is sufficient; no follow-up actions are"
            " required."
        )

      logger.info(
          "Synthesizing the final validation result from collected learnings."
      )
      all_learnings = self.tree_manager.get_all_learnings()
      if all_learnings:
        final_summary = await self._synthesize_final_result(all_learnings)
      else:
        final_summary = {
            "summary": f"Unable to verify details about: {event_summary}",
            "category": "Insufficient_Information",
        }

      self.tree_manager.complete_node(root_id)

      research_tree = self.tree_manager.build_tree_json()
      research_tree["task_type"] = "validation"

      research_tree["initial_source_urls_processed"] = (
          self.initial_source_urls_processed
      )
      research_tree["followup_search_queries_executed"] = (
          self.followup_search_queries_executed
      )
      research_tree["total_queries_executed"] = (
          self.initial_source_urls_processed
          + self.followup_search_queries_executed
      )

      logger.info(f"Validation task completed: {self.task_data['task_id']}")

      return final_summary, research_tree

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"Validation task failed: {str(e)}")
      error_summary = {
          "summary": f"Validation task failed: {str(e)}",
          "category": "Error",
      }
      research_tree = self.tree_manager.build_tree_json()
      return error_summary, research_tree

  async def _execute_action_plan(
      self, next_actions, root_id
  ):
    """2

    Args:
        next_actions:
        root_id: ID
    """
    logger.info(f"Planned next actions: {len(next_actions)}")

    for i, action in enumerate(next_actions):
      logger.info(
          f" {i+1}/{len(next_actions)}: {action.get('info_gap', 'Unknown')}"
      )

      if action.get("action_type") == "resolve_internally":

        query_description = f"Internal resolution: {action.get('info_gap')}"
        node_id = self.tree_manager.create_node(
            query_description, parent_id=root_id, node_type="query"
        )

        learning = {
            "source_url": "internal_knowledge",
            "internal_resolution": {
                "info_gap": action.get("info_gap"),
                "resolved_answer": action.get("resolved_answer"),
                "reasoning": "Resolved using common knowledge",
            },
            "extraction_method": "internal_knowledge",
            "timestamp": datetime.now().isoformat(),
        }

        self.tree_manager.add_learnings(node_id, [learning])
        self.tree_manager.complete_node(node_id)
        logger.info(
            f"Internal resolution result: {action.get('resolved_answer')}"
        )

      elif action.get("action_type") == "search":

        query = action.get("query", "")

        if query:
          if self.followup_search_queries_executed >= self.query_budget:
            logger.info(
                "Follow-up query budget reached: %s/%s. Skipping remaining"
                " search actions.",
                self.followup_search_queries_executed,
                self.query_budget,
            )
            break
          self.followup_search_queries_executed += 1
          logger.info(
              "Executing planned search query %s/%s: %s",
              self.followup_search_queries_executed,
              self.query_budget,
              query,
          )

          if self.task_type == "validation":
            node_id = await self._process_validation_query(query, root_id)
          else:
            node_id = await self._process_exploration_query(query, root_id)
        else:
          logger.warning(f"Unsupported internal action payload: {action}")
      else:
        logger.warning(
            f"Skipping unsupported action type: {action.get('action_type')}"
        )

  async def investigate_exploration_task(self):
    """Returns:

    Tuple of (final_summary, research_tree)
    """
    logger.info(f"Starting exploration task: {self.task_data['task_id']}")

    peak_date = self.peak_date.strftime("%Y-%m-%d")
    root_question = (
        f"Investigate the cause of search spike for '{self.keywords[0]}' on"
        f" {peak_date}"
    )
    root_id = self.tree_manager.create_node(root_question, node_type="root")

    try:

      exploration_queries = self._generate_exploration_queries()

      all_learnings = []
      for query in exploration_queries[: self.query_budget]:
        logger.info(f"Executing exploration query: {query}")

        node_id = await self._process_exploration_query(query, root_id)

        if node_id in self.tree_manager.nodes:
          node_learnings = self.tree_manager.nodes[node_id].get("learnings", [])
          all_learnings.extend(node_learnings)

      if all_learnings:
        final_summary = self._synthesize_exploration_results(all_learnings)
      else:
        final_summary = {
            "summary": f"No clear cause found for search spike on {peak_date}",
            "category": "Unknown",
        }

      self.tree_manager.complete_node(root_id)

      research_tree = self.tree_manager.build_tree_json()
      research_tree["task_type"] = "exploration"
      research_tree["total_queries_executed"] = len(
          exploration_queries[: self.query_budget]
      )

      logger.info(f"Exploration task completed: {self.task_data['task_id']}")

      return final_summary, research_tree

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"Exploration task failed: {str(e)}")
      error_summary = {
          "summary": f"Exploration task failed: {str(e)}",
          "category": "Error",
      }
      research_tree = self.tree_manager.build_tree_json()
      return error_summary, research_tree

  def _generate_validation_queries(self):
    """ """
    event_summary = self.event_data.get("event_summary", "")
    announcement_date = self.event_data.get("announcement_date", "")

    queries = []

    if event_summary:
      queries.append(f"{event_summary} verification facts")

    if announcement_date:
      queries.append(
          f"{self.keywords[0]} {announcement_date} official announcement"
      )

    queries.append(
        f"{self.keywords[0]} details specifications {announcement_date}"
    )

    queries.append(
        f"{self.keywords[0]} news multiple sources {announcement_date}"
    )

    return queries

  def _generate_exploration_queries(self):
    """ """
    keyword = self.keywords[0]
    peak_date = self.peak_date.strftime("%Y-%m-%d")
    peak_month_year = self.peak_date.strftime("%B %Y")

    queries = [
        f"{keyword} news {peak_date}",
        f"{keyword} announcement {peak_month_year}",
        f"what happened to {keyword} {peak_date}",
        f"{keyword} trending {peak_month_year}",
        f"{keyword} events {peak_date}",
    ]

    return queries

  async def _process_validation_query(self, query, parent_id):
    """Args:

    query:
    parent_id: ID
    """

    return await self._process_node(query, parent_id)

  async def _process_exploration_query(self, query, parent_id):
    """ """
    return await self._process_node(query, parent_id)

  @llm_retry_on_429(max_retries=5)
  async def _synthesize_final_result(self, learnings):
    """- verified_statements"""

    from datetime import datetime as dt_now

    if not learnings:
      return {
          "summary": "No validation information found",
          "category": "unverified",
      }

    all_confirmed_statements = []
    all_anticipated_statements = []
    content_dates = []
    publish_dates = []
    internal_resolutions = []

    earliest_learning = None
    earliest_publish_date = None

    for learning in learnings:

      if "verification_report" in learning:
        report = learning["verification_report"]
        if "verified_statements" in report:
          for stmt in report["verified_statements"]:
            if stmt.get("status") == "Confirmed":
              all_confirmed_statements.append(stmt)
            elif stmt.get("status") == "Anticipated":
              all_anticipated_statements.append(stmt)

        if "overall_timing" in report:
          timing = report["overall_timing"]
          if timing.get("content_date"):
            content_dates.append(timing["content_date"])
          if timing.get("publish_date"):
            current_publish_date = timing["publish_date"]
            publish_dates.append(current_publish_date)

            try:
              current_date_obj = datetime.strptime(
                  current_publish_date, "%Y-%m-%d"
              )
              if (
                  earliest_publish_date is None
                  or current_date_obj
                  < datetime.strptime(earliest_publish_date, "%Y-%m-%d")
              ):
                earliest_publish_date = current_publish_date
                earliest_learning = learning
            except (ValueError, TypeError):

              pass

      if "internal_resolution" in learning:
        internal_resolutions.append(learning["internal_resolution"])

    if not all_confirmed_statements and not internal_resolutions:

      supplementary_learnings = []
      for learning in learnings:

        if (
            "verification_report" not in learning
            and "event_summary" in learning
            and learning.get("event_summary")
            and learning.get("event_summary").strip() != ""
            and learning.get("event_summary") != "Not relevant"
        ):
          supplementary_learnings.append(learning)

      if supplementary_learnings:
        logger.info(
            "Supplementary learnings identified:"
            f" {len(supplementary_learnings)}"
        )

        recalibration_config = self.config.get("event_sourcing", {}).get(
            "recalibration", {}
        )
        if recalibration_config.get("enabled", False):
          logger.info("Recalibration is enabled; starting follow-up review.")

          original_claim = self.event_data.get("event_summary", "Unknown event")
          time_range = self.config.get("time_range", {})
          start_date = time_range.get("start", "2024-01-01")
          end_date = time_range.get("end", "2024-12-31")

          recalibration_result = await recalibrate_event_claim_with_search(
              original_claim=original_claim,
              start_date=start_date,
              end_date=end_date,
              config=self.config,
          )

          logger.info(
              "Running claim recalibration for original claim:"
              f" {original_claim[:50]}..."
          )
          logger.info(
              f"Claim recalibration time window: {start_date} to {end_date}"
          )
          logger.debug(
              f"Claim recalibration initial payload: {recalibration_result}"
          )

          root_node_id = self.tree_manager.root_id
          recalibration_log = {
              "recalibration_attempt": {
                  "original_claim": original_claim,
                  "time_window": f"{start_date} to {end_date}",
                  "result": recalibration_result,
                  "timestamp": dt_now.now().isoformat(),
              }
          }
          self.tree_manager.add_learnings(root_node_id, [recalibration_log])

          if (
              recalibration_result.get("recalibration_status")
              == "recalibrated_and_verified"
          ):
            event = recalibration_result.get("event")
            if event and event.get("source_urls"):
              new_urls = event.get("source_urls", [])
              logger.info(
                  f"Claim recalibration identified {len(new_urls)} new URLs for"
                  " follow-up review."
              )
              for i, url in enumerate(new_urls):
                logger.debug(f"Claim recalibration URL {i}: {url}")
              logger.info(
                  f"Existing source URL count: {len(event['source_urls'])} URLs"
              )

              new_query = event.get("event_summary", original_claim)
              self.tree_manager.update_node_query(root_node_id, new_query)

              new_learnings = []
              for url in event["source_urls"]:
                try:
                  logger.info(f"URL: {url}")
                  processed_node_id = await self._process_url_directly(
                      url, root_node_id
                  )
                  if processed_node_id:

                    logger.debug(
                        "Processed recalibration URL node id:"
                        f" {processed_node_id}"
                    )
                    logger.debug(
                        "Current research tree node ids:"
                        f" {list(self.tree_manager.nodes.keys())}"
                    )

                    if processed_node_id in self.tree_manager.nodes:
                      node_learnings = self.tree_manager.nodes[
                          processed_node_id
                      ]["learnings"]
                      logger.info(
                          f"Collected {len(node_learnings)} learnings from"
                          f" recalibration node {processed_node_id}."
                      )
                      new_learnings.extend(node_learnings)
                    else:
                      logger.error(
                          "Processed recalibration node was not found:"
                          f" {processed_node_id}"
                      )
                except ProviderSemanticError:
                  raise
                except Exception as e:
                  logger.error(f"URL {url}: {e}")

              if new_learnings:
                logger.info(
                    "New learnings extracted during recalibration:"
                    f" {len(new_learnings)}"
                )

                return await self._synthesize_final_result(new_learnings)
              else:
                logger.warning(
                    "Claim recalibration returned URLs, but no learnings were"
                    " extracted from them."
                )
            else:
              logger.warning(
                  "Claim recalibration returned an event without usable source"
                  " URLs."
              )
          else:

            failure_status = recalibration_result.get(
                "recalibration_status", "unknown"
            )
            logger.warning(
                "Claim recalibration did not verify the event. Status:"
                f" {failure_status}"
            )
            logger.debug(f"Claim recalibration payload: {recalibration_result}")
            logger.info("Stage transition complete.")

        logger.info(
            "The original claim remained unverified after recalibration."
        )
        return {
            "summary": (
                "The original claim could not be verified by any high-quality"
                " sources."
            ),
            "category": "unverified",
        }
      else:

        return {
            "summary": (
                "The original claim could not be verified by any sources or"
                " internal knowledge."
            ),
            "category": "unverified",
        }

    try:

      api_key = os.environ.get("GEMINI_API_KEY")
      if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

      client = _create_gemini_client(api_key)

      model_name = self.config.get("llm_models", {}).get(
          "final_synthesis", "gemini-3.5-flash"
      )
      logger.info(
          "Configured final synthesis model setting: "
          f"{self.config.get('llm_models', {}).get('final_synthesis', 'not configured')}"
      )
      logger.info(f"Using model for final synthesis: {model_name}")

      timing_evidence_lines = []
      for learning in learnings:
        if "verification_report" in learning:
          report = learning["verification_report"]
          if "overall_timing" in report:
            timing = report["overall_timing"]
            source_url = learning.get("source_url", "Unknown Source")
            publish_date = timing.get("publish_date", "N/A")
            content_date = timing.get("content_date", "N/A")
            timing_evidence_lines.append(
                f"- Source: {source_url} reported Publish Date: {publish_date},"
                f" Content Date: {content_date}"
            )
      timing_summary_detailed = (
          "\n".join(timing_evidence_lines)
          if timing_evidence_lines
          else "No specific date information found in sources."
      )

      unique_content_dates = sorted(list(set(content_dates)))
      unique_publish_dates = sorted(list(set(publish_dates)))
      confirmed_facts = "\n".join([
          f"- {stmt['statement']} (Quote:"
          f" {stmt.get('supporting_quote', 'N/A')})"
          for stmt in all_confirmed_statements
      ])
      anticipated_facts = "\n".join([
          f"- {stmt['statement']} (Quote:"
          f" {stmt.get('supporting_quote', 'N/A')})"
          for stmt in all_anticipated_statements
      ])
      internal_facts = "\n".join([
          f"- {res['info_gap']}: {res['resolved_answer']}"
          for res in internal_resolutions
      ])

      synthesis_prompt = f"""You are an Information Integration Specialist. Your task is to produce the final, authoritative version of an event by reviewing all provided evidence. Your summary must:

- Correct and enrich the original claim with additional verified details.
- For factual or scheudld event, prioritize the most authoritative sources.
- For subjective analyses or predictions, explicitly include multiple credible viewpoints, if available. Clearly acknowledging any conflicting or uncertain claims along with their sources, if available
- Accurately adjudicate or revise the event's announcement date (the date the news was published) and occurrence date (the actual date of the event), using web search if necessary to ensure accuracy.
-If you are unsure, use NA and avoid making up information.

**1. Original Event Claim:**
{self.event_data.get('event_summary', 'Unknown event')}

**2. Detailed Factual Evidence:**
**Confirmed Facts:**
{confirmed_facts if confirmed_facts else "None"}

**Anticipated/Planned Facts:**
{anticipated_facts if anticipated_facts else "None"}

**Internal Knowledge Resolutions:**
{internal_facts if internal_facts else "None"}

**3. Detailed Timing Evidence from Sources:**
{timing_summary_detailed}

**4. Intial Date:**
The following dates are preliminary findings and do not represent 100% accuracy. Please select the most reasonable date based on the content and use online search tools if necessary.
- **All Publish Dates Found:** {unique_publish_dates if unique_publish_dates else "None"}
- **All Content Dates Found:** {unique_content_dates if unique_content_dates else "None"}

**Your Final Task:**
Respond with ONLY a single JSON object. Do not add any text, explanations, or markdown formatting before or after the JSON block.

**JSON Output Format:**
{{
"final_summary_text": "<Your comprehensive summary here. This text should be well-written, accurate, detailed and reflect your final decision on the dates.>",
"authoritative_dates": {{
"announcement_date": "<The single, most credible YYYY-MM-DD publish date of content. If none, use NA.>",
"occurrence_date": "<The single, most credible YYYY-MM-DD date when the event actually took place. If none, use NA.>"  }},
"reasoning_for_date_choice": "<A brief, one-sentence explanation for your date selection. e.g., 'Chose the earliest publish date from a primary news source.'>"
}}
"""

      contents = [
          types.Content(
              role="user", parts=[types.Part.from_text(text=synthesis_prompt)]
          )
      ]

      tools = [types.Tool(googleSearch=types.GoogleSearch())]

      generate_content_config = types.GenerateContentConfig(
          max_output_tokens=65000, tools=tools
      )

      logger.info("Prepared final synthesis prompt.")

      logger.info("Sending final synthesis request to the LLM.")
      response_chunks = []

      for chunk in client.models.generate_content_stream(
          model=model_name,
          contents=contents,
          config=generate_content_config,
      ):
        if chunk.text:
          response_chunks.append(chunk.text)

      response_text = "".join(response_chunks).strip()

      final_summary = ""
      timing_info = {}
      category = "unverified"  # Default category
      reasoning = "No valid JSON returned from LLM."  # Default reasoning

      try:

        if response_text.startswith("```json"):
          response_text = response_text[7:]
        if response_text.endswith("```"):
          response_text = response_text[:-3]
        response_text = response_text.strip()

        parsed_json = json.loads(response_text)

        final_summary = parsed_json.get(
            "final_summary_text", "Summary not generated by LLM."
        )
        authoritative_dates = parsed_json.get("authoritative_dates", {})
        timing_info["announcement_date"] = authoritative_dates.get(
            "announcement_date"
        )
        timing_info["occurrence_date"] = authoritative_dates.get(
            "occurrence_date"
        )
        reasoning = parsed_json.get(
            "reasoning_for_date_choice", "No reasoning provided."
        )

        if final_summary and final_summary != "Summary not generated by LLM.":
          category = "verified"
          logger.info("Status diagnostics: synthesis status: verified by LLM")
        else:
          logger.warning(
              "Status diagnostics: synthesis status: unverified by LLM"
          )

      except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse LLM JSON response: {e}. Falling back to raw text"
            " output."
        )
        logger.debug(f"Invalid LLM Response: {response_text}")

        final_summary = response_text
        if unique_publish_dates:
          timing_info["announcement_date"] = min(unique_publish_dates)
        if unique_content_dates:
          timing_info["occurrence_date"] = (
              min(unique_content_dates) if unique_content_dates else None
          )

        if all_confirmed_statements:
          category = "verified"
          logger.info(
              "Status diagnostics: (Fallback) verified:"
              f" {len(all_confirmed_statements)}"
          )

      supporting_quotes = [
          stmt.get("supporting_quote", "")
          for stmt in all_confirmed_statements
          if stmt.get("supporting_quote")
      ]

      return {
          "summary": final_summary,
          "category": category,
          "timing_info": timing_info,
          "supporting_quotes": supporting_quotes[:3],
          "confirmed_statements_count": len(all_confirmed_statements),
          "anticipated_statements_count": len(all_anticipated_statements),
          "internal_resolutions_count": len(internal_resolutions),
          "synthesis_reasoning": reasoning,
      }

    except ProviderSemanticError:
      raise
    except Exception as e:
      if is_llm_rate_limit_error(str(e)):
        raise e
      logger.error(f"Exploration synthesis failed: {str(e)}", exc_info=True)

      if all_confirmed_statements:
        category = "verified"
        simple_summary = "; ".join(
            [stmt["statement"] for stmt in all_confirmed_statements[:3]]
        )
        return {
            "summary": f"Event confirmed (synthesis failed): {simple_summary}",
            "category": category,
        }
      else:
        return {
            "summary": (
                "Event details resolved through internal knowledge, but"
                f" synthesis failed: {str(e)}"
            ),
            "category": "unverified",
        }

  def _synthesize_exploration_results(self, learnings):
    """ """
    if not learnings:
      return {
          "summary": "No relevant events found for the search spike",
          "category": "Unknown",
      }

    best_learning = max(
        learnings, key=lambda x: len(x.get("event_summary", ""))
    )

    return {
        "summary": best_learning.get(
            "event_summary", "Search spike cause identified"
        ),
        "category": "Discovered",
    }

  async def investigate(self):
    """Research flow helper."""
    if self.task_type == "validation":
      return await self.investigate_validation_task()
    elif self.task_type == "exploration":
      return await self.investigate_exploration_task()
    else:
      raise ValueError(f"Unsupported task_type: {self.task_type}")

  async def _process_node(
      self,
      query,
      parent_id = None,
      node_id = None,
      use_date_limit = None,
      sort_by_date = False,
  ):
    """Process a single research node and attach it to the tree."""

    if self.task_type == "validation":
      use_date_limit = True
    elif use_date_limit is None:
      use_date_limit = False

    if node_id is None:
      node_id = self.tree_manager.create_node(
          query, node_type="query", parent_id=parent_id
      )
    logger.info(f"Processing node: {query}")

    try:

      search_items, search_metadata = await self._google_search(
          query, self.iterative_crawl_count, use_date_limit, sort_by_date
      )

      if search_metadata:
        self.tree_manager.add_metadata(
            node_id, {"search_time_range": search_metadata}
        )

      if not search_items:
        logger.info(f"No URLs found for query: {query}")
        self.tree_manager.complete_node(node_id)
        return node_id

      urls = [item.get("link", "") for item in search_items if item.get("link")]

      learnings = []
      sources = []

      try:
        async with AsyncWebCrawler(verbose=False) as crawler:
          crawl_tasks = [crawler.arun(url=url) for url in urls]
          crawl_results = await asyncio.gather(
              *crawl_tasks, return_exceptions=True
          )

        summary_tasks = []

        for i, result in enumerate(crawl_results):
          url = urls[i]
          if (
              isinstance(result, Exception)
              or not result
              or not getattr(result, "success", False)
          ):
            logger.warning(f"Skipping failed/empty crawl: {url}")
            sources.append({
                "url": url,
                "title": "Unknown Title",
                "status": "crawl_failed",
            })
            continue

          cleaned = clean_crawl_evidence(result, self.config)
          clean_meta = cleaning_metadata(cleaned)
          title = getattr(result, "title", None)
          metadata = getattr(result, "metadata", {}) or {}
          if not title:
            title = metadata.get("title", "Unknown Title")

          source_record = {
              "url": url,
              "title": title,
              "status": (
                  "cleaned"
                  if clean_meta["cleaning_status"] == "success"
                  else "content_cleaning_failed"
              ),
              **clean_meta,
          }
          sources.append(source_record)

          if clean_meta["cleaning_status"] != "success":
            logger.info(
                "Skipping source after evidence cleaning failed: %s (%s)",
                url,
                clean_meta.get("cleaning_message"),
            )
            continue

          task = get_single_source_summary(
              cleaned["text"], url, query, self.keywords, self.config
          )
          summary_tasks.append(task)

        if summary_tasks:
          individual_summaries = await asyncio.gather(*summary_tasks)
          valid_summaries = [
              s
              for s in individual_summaries
              if s and s.get("event_summary") != "Not relevant"
          ]

          for summary in valid_summaries:
            if summary.get("event_summary"):
              learnings.append(summary)

          logger.info(
              f"Extracted {len(learnings)} learnings from crawling {len(urls)}"
              f" sources for query: {query}"
          )

      except ProviderSemanticError:
        raise
      except Exception as crawl_error:
        logger.warning(
            f"Crawling failed for query '{query}': {str(crawl_error)}"
        )

      if not learnings and search_items:
        logger.warning(
            "Snippet fallback is disabled for query '%s'. Search results were"
            " found, but no crawl-derived learnings were produced.",
            query,
        )

      if learnings:
        self.tree_manager.add_learnings(node_id, learnings)
      if sources:
        self.tree_manager.add_sources(node_id, sources)

      self.tree_manager.complete_node(node_id)
      return node_id

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"Error processing node {query}: {str(e)}")
      self.tree_manager.complete_node(node_id)
      return node_id

  async def _google_search(
      self,
      query,
      num_results,
      use_date_limit = False,
      sort_by_date = False,
  ):
    """Google"""

    logger.debug(f"Search diagnostics: _google_search called with parameters:")
    logger.debug(f"Search diagnostics:   query: '{query}'")
    logger.debug(f"Search diagnostics:   num_results: {num_results}")
    logger.debug(f"Search diagnostics:   use_date_limit: {use_date_limit}")
    logger.debug(f"Search diagnostics:   sort_by_date: {sort_by_date}")
    logger.debug(f"Search diagnostics:   task_type: {self.task_type}")
    logger.debug(f"Search diagnostics:   peak_date: {self.peak_date}")

    try:

      api_key = os.environ.get("CUSTOMSEARCH_API_KEY")
      cx_key = os.environ.get("CUSTOMSEARCH_CX_KEY")

      if not api_key:
        logger.error(
            "Search diagnostics: CUSTOMSEARCH_API_KEY not found in environment"
            " variables - this will cause search failure"
        )
      else:
        logger.debug(
            "Search diagnostics: CUSTOMSEARCH_API_KEY successfully loaded from"
            " environment"
        )

      if not cx_key:
        logger.error(
            "Search diagnostics: CUSTOMSEARCH_CX_KEY not found in environment"
            " variables - this will cause search failure"
        )
      else:
        logger.debug(
            "Search diagnostics: CUSTOMSEARCH_CX_KEY successfully loaded from"
            " environment"
        )

      if not api_key or not cx_key:
        logger.error(
            "Search diagnostics: Missing required Custom Search credentials"
        )
        raise ProviderConfigurationError(
            "customsearch",
            "Both CUSTOMSEARCH_API_KEY and CUSTOMSEARCH_CX_KEY are required for"
            " Google Custom Search.",
        )

      url = "https://www.googleapis.com/customsearch/v1"

      logger.debug(f"Search diagnostics: Search query: '{query}'")
      search_query = query

      params = {
          "key": api_key,
          "cx": cx_key,
          "q": search_query,
          "num": num_results,
      }

      if self.task_type == "validation":
        start_date, end_date = self._get_validation_search_window()
      else:

        time_range = self.config.get("time_range", {})
        start_date = time_range.get("start")
        end_date = time_range.get("end")

      if use_date_limit and start_date and end_date:

        start_formatted = start_date.replace("-", "")  # 2024-01-01 -> 20240101
        end_formatted = end_date.replace("-", "")  # 2024-03-31 -> 20240331
        # Google Custom Search JSON API uses sort=date:r:YYYYMMDD:YYYYMMDD for absolute date range filtering
        params["sort"] = f"date:r:{start_formatted}:{end_formatted}"

        logger.info(
            f"Date-constrained search window: {start_date} to {end_date}"
        )
        search_metadata = {
            "date_constraint_mode": "date_restricted",
            "date_range": f"{start_date} to {end_date}",
        }
      elif sort_by_date:

        params["sort"] = "date"
        logger.info("Stage transition complete.")
        search_metadata = {"date_constraint_mode": "sorted_by_date"}
      else:

        logger.info("Recalibration is enabled; starting follow-up review.")
        search_metadata = {"date_constraint_mode": "no_limit"}

      logger.info(f"Search diagnostics: Final API request parameters:")
      logger.info(f"Search diagnostics:   URL: {url}")
      logger.info(
          "Search diagnostics:   API Key:"
          f" {'***PRESENT***' if api_key else 'MISSING'}"
      )
      logger.info(
          "Search diagnostics:   CX Key:"
          f" {'***PRESENT***' if cx_key else 'MISSING'}"
      )
      if "tbs" in params:
        logger.info(
            f"Search diagnostics:   tbs (date restriction): {params['tbs']}"
        )
      else:
        logger.info(f"no tbs available")
      for key, value in params.items():
        if key not in ["key", "cx"]:
          logger.info(f"Search diagnostics:   {key}: {value}")

      response = requests.get(url, params=params, timeout=10)
      if not response.ok:
        classification = _classify_customsearch_error(response)
        detail = f"status={response.status_code};body={response.text[:300]}"
        if classification == "daily":
          raise WaitingNextDayQuotaError("customsearch", detail)
        if classification == "transient":
          raise PendingRetryProviderError(
              "customsearch", "customsearch_rate_limited", detail
          )
        response.raise_for_status()

      logger.debug(
          f"Search diagnostics: HTTP Response Status: {response.status_code}"
      )
      logger.debug(
          f"Search diagnostics: HTTP Response Headers: {dict(response.headers)}"
      )
      logger.debug(
          f"Search diagnostics: HTTP Response Body: {response.text[:1000]}..."
      )

      search_results = response.json()

      logger.debug(
          f"Search diagnostics: Parsed JSON keys: {list(search_results.keys())}"
      )
      if "items" in search_results:
        logger.info(
            f"Search diagnostics: Found {len(search_results['items'])} items in"
            " API response"
        )
      else:
        logger.warning(
            f"Search diagnostics: API response successful but missing 'items'"
            f" field - possible empty result or invalid query"
        )
        logger.debug(
            f"Search diagnostics: Full response structure: {search_results}"
        )

      items = []
      if "items" in search_results:
        items = search_results["items"]

      logger.info(f"Found {len(items)} URLs for search query: {search_query}")

      logger.debug(
          f"Search diagnostics: Returning {len(items)} items and search"
          f" metadata: {search_metadata}"
      )

      return items, search_metadata

    except ProviderSemanticError:
      raise
    except Exception as e:
      date_constraint_mode = (
          search_metadata.get("date_constraint_mode", "unknown")
          if isinstance(locals().get("search_metadata"), dict)
          else "unknown"
      )
      logger.error(
          "Search diagnostics: Google search failed for query '%s' "
          "(date_constraint_mode=%s, error_type=%s)",
          query,
          date_constraint_mode,
          type(e).__name__,
      )
      logger.error(f"Error in Google search for query '{query}': {str(e)}")
      return [], {}

  def _get_validation_search_window(self):
    """Calculate search window for validation tasks.

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """

    announcement_date = self.event_data.get("announcement_date")
    if announcement_date:
      try:
        anchor_date = datetime.strptime(announcement_date, "%Y-%m-%d")
        start_date = (anchor_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (anchor_date + timedelta(days=7)).strftime("%Y-%m-%d")

        logger.info(
            f"Announcement date window: {announcement_date} +/- 7 days ="
            f" [{start_date}, {end_date}]"
        )
        return start_date, end_date
      except ValueError:
        logger.warning(f"Invalid announcement_date format: {announcement_date}")

    # Fallback to the global config time range
    time_range = self.config.get("time_range", {})
    start_date = time_range.get("start", "2024-01-01")
    end_date = time_range.get("end", "2024-12-31")

    logger.info(f"Fallback to global time window: [{start_date}, {end_date}]")
    return start_date, end_date

  async def _process_url_directly(self, url, parent_id):
    """Process one source URL directly and attach the result to the research tree.

    Args:
        url: URL
        parent_id: ID

    Returns:
        ID
    """

    node_id = self.tree_manager.create_node(
        f"Direct URL processing: {url}", parent_id=parent_id
    )

    try:

      logger.info(f"URL: {url}")

      async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        if result.success:

          logger.debug(f"Verification diagnostics: Crawl successful for: {url}")
          cleaned = clean_crawl_evidence(result, self.config)
          clean_meta = cleaning_metadata(cleaned)
          logger.debug(
              "Verification diagnostics: Evidence cleaning status=%s method=%s"
              " raw_length=%s clean_length=%s",
              clean_meta["cleaning_status"],
              clean_meta["cleaning_method"],
              clean_meta["raw_length"],
              clean_meta["clean_length"],
          )

          if clean_meta["cleaning_status"] != "success":
            logger.info(f"Evidence cleaning rejected URL: {url}")
            self.tree_manager.add_sources(
                node_id,
                [{
                    "url": url,
                    "status": "content_cleaning_failed",
                    **clean_meta,
                }],
            )
            return node_id

          cleaned_text = cleaned["text"]
          content_lower = cleaned_text.lower()[:1000]

          metadata = getattr(result, "metadata", {}) or {}
          title_lower = (metadata.get("title") or "").lower()

          error_keywords = [
              "page not found",
              "404 not found",
              "404 error",
              "access denied",
              "permission denied",
              "login required",
              "page does not exist",
              "content not available",
              "server error",
              "temporarily unavailable",
          ]

          contains_error = any(
              keyword in content_lower or keyword in title_lower
              for keyword in error_keywords
          )

          if contains_error:
            logger.info(f"Layer 1 filter rejected URL: {url}")
            self.tree_manager.add_sources(
                node_id,
                [{
                    "url": url,
                    "status": "blocked_by_keyword_filter",
                    **clean_meta,
                }],
            )
            return node_id

          event_claim = self.event_data.get("event_summary", "")

          logger.debug(
              "Verification diagnostics: Claim being sent for verification:"
              f" '{event_claim}'"
          )

          max_len = self.config.get("event_sourcing", {}).get(
              "max_content_length", 20000
          )

          effective_length = min(len(cleaned_text), max_len)
          truncated_evidence = cleaned_text[:effective_length]

          logger.debug(
              f"Evidence length before truncation={len(cleaned_text)},"
              f" max_allowed={max_len}, effective_length={effective_length}"
          )

          verification_result = await verify_claim_with_evidence(
              event_claim=event_claim,
              source_evidence=truncated_evidence,
              config=self.config,
          )

          logger.debug(
              "Verification diagnostics: Raw result returned from LLM"
              f" verifier: {verification_result}"
          )

          page_status = verification_result.get("page_status", "unknown")

          if page_status != "valid_content":
            logger.info(
                f"Layer 3 filter rejected URL with page_status '{page_status}':"
                f" {url}"
            )

            learning = {
                "source_url": url,
                "verification_report": verification_result,
                "extraction_method": "invalid_page_filtered",
                "evidence_cleaning": clean_meta,
                "timestamp": datetime.now().isoformat(),
            }
            self.tree_manager.add_learnings(node_id, [learning])
            self.tree_manager.add_sources(
                node_id,
                [{
                    "url": url,
                    "status": f"blocked_by_page_status_{page_status}",
                    **clean_meta,
                }],
            )
            return node_id

          learning = {
              "source_url": url,
              "verification_report": verification_result,
              "extraction_method": "claim_verification",
              "evidence_cleaning": clean_meta,
              "timestamp": datetime.now().isoformat(),
          }

          self.tree_manager.add_learnings(node_id, [learning])
          self.tree_manager.add_sources(
              node_id, [{"url": url, "status": "success", **clean_meta}]
          )

          confirmed_statements = [
              stmt
              for stmt in verification_result.get("verified_statements", [])
              if stmt.get("status") == "Confirmed"
          ]
          is_verified = len(confirmed_statements) > 0

          logger.info(
              "Verification result for URL %s: verified=%s"
              " confirmed_statements=%s",
              url,
              is_verified,
              len(confirmed_statements),
          )
        else:
          logger.warning(f"URL: {url}")
          self.tree_manager.add_sources(
              node_id, [{"url": url, "status": "crawl_failed"}]
          )

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"URL {url}: {str(e)}")
      self.tree_manager.add_sources(
          node_id, [{"url": url, "status": "error", "error": str(e)}]
      )

    self.tree_manager.complete_node(node_id)
    return node_id

  @llm_retry_on_429(max_retries=5)
  async def _evaluate_and_plan_next_steps(self):
    """Evaluate whether the current evidence is sufficient and plan next actions.

    Returns:
        is_sufficientnext_actions
    """

    max_actions = self.config.get("event_sourcing", {}).get(
        "max_evaluation_actions", 5
    )

    all_learnings = self.tree_manager.get_all_learnings()

    if not all_learnings:
      return {
          "is_sufficient": False,
          "next_actions": [{
              "info_gap": "No initial information collected",
              "is_common_knowledge": False,
              "action_type": "search",
              "query": (
                  f"{self.config['target_variable']}"
                  f" {self.event_data.get('event_summary', 'event')} details"
              ),
          }],
      }

    all_verified_statements = []
    overall_timings = []

    for learning in all_learnings:
      if "verification_report" in learning:
        report = learning["verification_report"]
        if "verified_statements" in report:
          all_verified_statements.extend(report["verified_statements"])
        if "overall_timing" in report:
          overall_timings.append(report["overall_timing"])

    if not all_verified_statements:
      return {
          "is_sufficient": False,
          "next_actions": [{
              "info_gap": "No verified statements found in sources",
              "is_common_knowledge": False,
              "action_type": "search",
              "query": (
                  f"{self.config['target_variable']} official announcement"
              ),
          }],
      }

    try:
      original_claim = self.event_data.get("event_summary", "Unknown event")

      confirmed_statements = [
          stmt
          for stmt in all_verified_statements
          if stmt.get("status") == "Confirmed"
      ]
      anticipated_statements = [
          stmt
          for stmt in all_verified_statements
          if stmt.get("status") == "Anticipated"
      ]

      statements_summary = f"""
Confirmed Facts ({len(confirmed_statements)}):
{chr(10).join([f"- {stmt['statement']}" for stmt in confirmed_statements])}

Anticipated/Planned Facts ({len(anticipated_statements)}):
{chr(10).join([f"- {stmt['statement']}" for stmt in anticipated_statements])}

Overall Timing Information:
{chr(10).join([f"- Content Date: {timing.get('content_date', 'Unknown')}, Publish Date: {timing.get('publish_date', 'Unknown')}" for timing in overall_timings if timing])}
"""

      api_key = os.environ.get("GEMINI_API_KEY") or os.getenv(
          "GEMINI_API_KEY_FALLBACK", ""
      )
      client = _create_gemini_client(api_key)

      model_name = self.config.get("llm_models", {}).get(
          "information_sufficiency_evaluator", "gemini-3.5-flash"
      )
      logger.info(
          "information_sufficiency_evaluator:"
          f" {self.config.get('llm_models', {}).get('information_sufficiency_evaluator', 'not configured')}"
      )
      logger.info(
          f"Using model for information sufficiency evaluation: {model_name}"
      )

      time_range = self.config.get("time_range", {})
      start_date = time_range.get("start", "2024-01-01")
      end_date = time_range.get("end", "2024-12-31")

      prompt = f"""You are a research strategist analyzing information gaps and planning next steps.

**IMPORTANT ACTION LIMIT**: Please limit your next_actions to a maximum of {max_actions} items. Focus on the most critical information gaps that need to be addressed. Each action should be high-quality and targeted.

**Original Event Claim:**
{original_claim}

**Currently Verified Information:**
{statements_summary}

**Your Task:**
1. **Analyze completeness**: Compare verified information against the original claim
2. **Identify information gaps**: List missing or unconfirmed facts
3. **Plan next actions**: For each gap, determine if it's common knowledge or requires search

For each information gap, classify as:
- **Common knowledge**: Facts that can be resolved internally (e.g., "Apple's fiscal Q1 is Oct-Dec")
- **Requires search**: Facts needing external verification

**JSON Output format:**
{{
  "is_sufficient": <true if all key facts are confirmed, false otherwise>,
  "next_actions": [
    {{
      "info_gap": "<description of missing information>",
      "is_common_knowledge": <true|false>,
      "action_type": "<resolve_internally|search>",
      "resolved_answer": "<answer if common knowledge, null otherwise>",
      "query": "<search query if action_type is search, null otherwise>"
    }}
  ]
}}

Please respond with ONLY the JSON object."""

      response = await client.aio.models.generate_content(
          model=model_name, contents=prompt
      )
      response_text = response.text.strip()

      if response_text.startswith("```json"):
        response_text = response_text[7:]
      if response_text.endswith("```"):
        response_text = response_text[:-3]
      response_text = response_text.strip()
      result = json.loads(response_text)

      if "is_sufficient" not in result:
        result["is_sufficient"] = False
      if "next_actions" not in result:
        result["next_actions"] = []

      return result

    except ProviderSemanticError:
      raise
    except Exception as e:
      if is_llm_rate_limit_error(str(e)):
        raise e
      logger.error(f"Error in evaluation and planning: {str(e)}")

      return {
          "is_sufficient": False,
          "next_actions": [{
              "info_gap": f"Evaluation failed: {str(e)}",
              "is_common_knowledge": False,
              "action_type": "search",
              "query": f"{self.config['target_variable']} details verification",
          }],
      }

  def _generate_supplementary_queries(
      self, next_actions
  ):
    """Args:

        next_actions:

    Returns:
    """
    event_summary = self.event_data.get("event_summary", "Unknown event")
    keyword = self.config["target_variable"]

    search_actions = []
    for action in next_actions:
      if action.get("action_type") == "search":
        search_actions.append(action)
      elif action.get("action_type") == "resolve_internally":

        learning = {
            "source_url": "internal_knowledge",
            "internal_resolution": {
                "info_gap": action.get("info_gap"),
                "resolved_answer": action.get("resolved_answer"),
            },
            "extraction_method": "internal_knowledge",
            "timestamp": datetime.now().isoformat(),
        }
        self.tree_manager.add_learnings(
            self.tree_manager.root_id, [learning]
        )  # Add to root node

    if not search_actions:
      search_actions = [
          {
              "query": f"{keyword} {event_summary} details",
              "use_date_limit": False,
          },
          {
              "query": f"{keyword} official announcement",
              "use_date_limit": False,
          },
      ]

    return search_actions[: self.query_budget]
