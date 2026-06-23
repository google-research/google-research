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

"""Event-Centric Pipeline:

pipeline.
"""

import asyncio
from datetime import datetime
import fcntl
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataset_agent.modules.llm_retry_handler import ProviderSemanticError
from dataset_agent.modules.module_04_events import IterativeResearcher, ResearchTreeManager
from dataset_agent.modules.module_05_initial_events import calculate_coverage_rate, get_initial_events_with_iterations, llm_discovery_with_coverage_supervision
from dataset_agent.modules.module_06_peak_discovery import get_significant_peaks_via_trends
from dataset_agent.modules.module_07_cross_validation import cross_validate_and_allocate, generate_task_summary
from dataset_agent.modules.task_runtime import set_terminal_state
import pandas as pd

logger = logging.getLogger(__name__)


def persist_workflow_outcome(
    config,
    workflow_outcome,
    total_events = None,
    verified_events = None,
):
  """Persists the workflow outcome and event counts to the JSON dataset file.

  Args:
    config: Pipeline configuration dictionary.
    workflow_outcome: String indicating the final workflow result.
    total_events: Optional total number of events processed.
    verified_events: Optional number of verified events.
  """
  try:
    final_dataset_path = config['paths']['final_dataset']
    json_path = (
        final_dataset_path.replace('.csv', '.json')
        if final_dataset_path.endswith('.csv')
        else final_dataset_path
    )
    if not os.path.exists(json_path):
      return

    with open(json_path, 'r+', encoding='utf-8') as f:
      fcntl.flock(f.fileno(), fcntl.LOCK_EX)
      try:
        f.seek(0)
        dataset = json.load(f)
        metadata = dataset.setdefault('metadata', {})
        processing = metadata.setdefault('processing_metadata', {})
        processing['workflow_outcome'] = workflow_outcome
        processing['last_updated'] = datetime.now().isoformat()
        if total_events is not None:
          processing['total_events'] = int(total_events)
        if verified_events is not None:
          processing['verified_events'] = int(verified_events)
        f.seek(0)
        f.truncate()
        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
      finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
  except Exception as e:
    logger.warning(f'Failed to persist workflow outcome: {str(e)}')


def initialize_dataset_files(config):
  """Initializes the output dataset files (CSV and JSON) based on configuration.

  Args:
    config: Pipeline configuration dictionary.
  """
  try:

    final_dataset_path = config['paths']['final_dataset']
    csv_path = final_dataset_path
    json_path = (
        final_dataset_path.replace('.csv', '.json')
        if final_dataset_path.endswith('.csv')
        else final_dataset_path.replace('.json', '.json')
    )

    os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)

    if csv_path.endswith('.csv'):

      empty_df = events_to_dataframe([])
      if empty_df.empty:

        columns = [
            'event_id',
            'discovery_channel',
            'event_summary',
            'announcement_date',
            'occurrence_date',
            'event_type',
            'validation_status',
            'peak_date',
            'peak_value',
            'correlation_score',
            'research_tree_file',
        ]
        empty_df = pd.DataFrame(columns=columns)

      empty_df.to_csv(csv_path, index=False, encoding='utf-8')
      logger.info(f'CSV: {csv_path}')

      unverified_csv_path = csv_path.replace('.csv', '_unverified.csv')
      empty_df.to_csv(unverified_csv_path, index=False, encoding='utf-8')
      logger.info(f'CSV: {unverified_csv_path}')

    keyword = config['target_variable']
    domain = config.get('domain', 'General Analysis')
    geography_str = config.get('geography_str', 'United States')
    time_range = config['time_range']

    initial_dataset = {
        'metadata': {
            'keyword': keyword,
            'domain': domain,
            'geography': geography_str,
            'time_range': time_range,
            'processing_metadata': {
                'total_events': 0,
                'llm_discovered': 0,
                'trends_discovered': 0,
                'verified_events': 0,
                'processing_timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
            },
        },
        'events': [],
    }

    with open(json_path, 'w', encoding='utf-8') as f:
      json.dump(initial_dataset, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f'JSON: {json_path}')

  except Exception as e:
    logger.error(f'Failed to initialize dataset files: {str(e)}')
    raise


def append_event_to_dataset(event_data, config):
  """Appends a single consolidated event record to the dataset files.

  Args:
    event_data: Dictionary containing the event data to append.
    config: Pipeline configuration dictionary.
  """
  try:
    final_dataset_path = config['paths']['final_dataset']
    csv_path = final_dataset_path
    json_path = (
        final_dataset_path.replace('.csv', '.json')
        if final_dataset_path.endswith('.csv')
        else final_dataset_path.replace('.json', '.json')
    )

    if csv_path.endswith('.csv'):

      event_df = events_to_dataframe([event_data])
      if not event_df.empty:
        validation_status = event_data.get('validation_status', 'unknown')

        if validation_status in ['verified', 'conflicting_reports']:

          event_df.to_csv(
              csv_path, mode='a', header=False, index=False, encoding='utf-8'
          )
          logger.info(
              f" {event_data.get('event_id', 'unknown')} CSV: {csv_path}"
          )
        else:

          unverified_csv_path = csv_path.replace('.csv', '_unverified.csv')
          event_df.to_csv(
              unverified_csv_path,
              mode='a',
              header=False,
              index=False,
              encoding='utf-8',
          )
          logger.info(
              f" {event_data.get('event_id', 'unknown')} CSV:"
              f' {unverified_csv_path}'
          )

    with open(json_path, 'r+', encoding='utf-8') as f:

      fcntl.flock(f.fileno(), fcntl.LOCK_EX)
      try:
        f.seek(0)
        dataset = json.load(f)

        dataset['events'].append(event_data)

        dataset['metadata']['processing_metadata']['total_events'] = len(
            dataset['events']
        )
        dataset['metadata']['processing_metadata']['llm_discovered'] = len(
            [e for e in dataset['events'] if e['discovery_channel'] == 'llm']
        )
        dataset['metadata']['processing_metadata']['trends_discovered'] = len(
            [e for e in dataset['events'] if e['discovery_channel'] == 'trends']
        )
        dataset['metadata']['processing_metadata']['verified_events'] = len([
            e for e in dataset['events'] if e['validation_status'] == 'verified'
        ])
        dataset['metadata']['processing_metadata'][
            'last_updated'
        ] = datetime.now().isoformat()

        verification_breakdown = calculate_verification_breakdown(
            dataset['events']
        )
        dataset['metadata']['processing_metadata'][
            'verification_breakdown'
        ] = verification_breakdown

        url_quality_stats = calculate_url_quality_stats(
            dataset['events'], config
        )
        dataset['metadata']['processing_metadata'][
            'url_quality_stats'
        ] = url_quality_stats

        f.seek(0)
        f.truncate()
        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)

        logger.info(
            f" {event_data.get('event_id', 'unknown')} JSON: {json_path}"
        )

      finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

  except Exception as e:
    logger.error(f'Failed to append event to dataset: {str(e)}')


def update_dataset_metadata(
    config,
    keyword,
    domain,
    geography_str,
    time_range,
):
  """Updates the global metadata and processing statistics in the JSON dataset.

  Args:
    config: Pipeline configuration dictionary.
    keyword: Target search variable or keyword.
    domain: Domain of analysis.
    geography_str: Geography string representation.
    time_range: Dictionary specifying start and end time range.
  """
  try:
    final_dataset_path = config['paths']['final_dataset']
    json_path = (
        final_dataset_path.replace('.csv', '.json')
        if final_dataset_path.endswith('.csv')
        else final_dataset_path.replace('.json', '.json')
    )

    if not os.path.exists(json_path):
      return

    with open(json_path, 'r+', encoding='utf-8') as f:
      fcntl.flock(f.fileno(), fcntl.LOCK_EX)
      try:
        f.seek(0)
        dataset = json.load(f)

        dataset['metadata']['keyword'] = keyword
        dataset['metadata']['domain'] = domain
        dataset['metadata']['geography'] = geography_str
        dataset['metadata']['time_range'] = time_range

        events = dataset.get('events', [])
        dataset['metadata']['processing_metadata']['total_events'] = len(events)
        dataset['metadata']['processing_metadata']['llm_discovered'] = len(
            [e for e in events if e['discovery_channel'] == 'llm']
        )
        dataset['metadata']['processing_metadata']['trends_discovered'] = len(
            [e for e in events if e['discovery_channel'] == 'trends']
        )
        dataset['metadata']['processing_metadata']['verified_events'] = len(
            [e for e in events if e['validation_status'] == 'verified']
        )
        dataset['metadata']['processing_metadata'][
            'last_updated'
        ] = datetime.now().isoformat()

        verification_breakdown = calculate_verification_breakdown(events)
        dataset['metadata']['processing_metadata'][
            'verification_breakdown'
        ] = verification_breakdown

        url_quality_stats = calculate_url_quality_stats(events, config)
        dataset['metadata']['processing_metadata'][
            'url_quality_stats'
        ] = url_quality_stats

        f.seek(0)
        f.truncate()
        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f'JSON output path: {json_path}')

      finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

  except Exception as e:
    logger.error(f'Failed to update dataset metadata: {str(e)}')


async def run_event_centric_pipeline(config):
  """Orchestrates the entire event-centric discovery and validation pipeline.

  Args:
    config: Pipeline configuration dictionary.

  Returns:
    Consolidated DataFrame of validated events, or None if pipeline fails.
  """
  logger.info('=' * 80)
  logger.info(' (Event-Centric Pipeline)')
  logger.info('=' * 80)

  current_stage = 'pipeline_start'
  try:

    keyword = config['target_variable']
    time_range = config['time_range']
    domain = config.get('domain', 'General Analysis')
    geography = config.get('geography', {})
    geography_str = config.get('geography_str', 'United States')

    logger.info(f'Keyword: {keyword}')
    logger.info(f"Time range: {time_range['start']} to {time_range['end']}")
    logger.info(f'Domain: {domain}')
    logger.info(f'Geography: {geography_str}')

    logger.info('Preparing dataset files.')
    initialize_dataset_files(config)

    current_stage = 'parallel_discovery'
    logger.info('\n' + '=' * 60)
    logger.info('Stage 1: Parallel Discovery')
    logger.info('=' * 60)

    initial_events, significant_peaks = await parallel_discovery_stage(
        keyword, time_range, domain, geography_str, geography, config
    )

    if not initial_events and not significant_peaks:
      logger.warning('No additional details were provided for this stage.')
      set_terminal_state(
          config,
          'no_events_found',
          'no_initial_events_or_peaks',
          current_stage,
          total_events=0,
          verified_events=0,
          safe_to_accept=False,
          workflow_outcome='no_events_found',
      )
      persist_workflow_outcome(
          config, 'no_events_found', total_events=0, verified_events=0
      )
      return None

    current_stage = 'cross_validation_and_task_allocation'
    logger.info('\n' + '=' * 60)
    logger.info('Stage 2: Cross-Validation and Task Allocation')
    logger.info('=' * 60)

    validation_tasks, exploration_tasks = cross_validation_and_allocation_stage(
        initial_events, significant_peaks, config
    )

    if not validation_tasks and not exploration_tasks:
      logger.warning('No additional details were provided for this stage.')
      set_terminal_state(
          config,
          'no_events_found',
          'no_validation_or_exploration_tasks',
          current_stage,
          total_events=0,
          verified_events=0,
          safe_to_accept=False,
          workflow_outcome='no_events_found',
      )
      persist_workflow_outcome(
          config, 'no_events_found', total_events=0, verified_events=0
      )
      return None

    current_stage = 'execution_and_consolidation'
    logger.info('\n' + '=' * 60)
    logger.info('Stage 3: Execution and Consolidation')
    logger.info('=' * 60)

    final_events = await execution_and_consolidation_stage(
        validation_tasks, exploration_tasks, keyword, config
    )

    if not final_events:
      logger.warning('No additional details were provided for this stage.')
      set_terminal_state(
          config,
          'no_events_found',
          'no_final_events_after_execution',
          current_stage,
          total_events=0,
          verified_events=0,
          safe_to_accept=False,
          workflow_outcome='no_events_found',
      )
      persist_workflow_outcome(
          config, 'no_events_found', total_events=0, verified_events=0
      )
      return None

    current_stage = 'finalize_dataset'
    logger.info('\n' + '=' * 60)
    logger.info('Stage transition complete.')
    logger.info('=' * 60)

    update_dataset_metadata(config, keyword, domain, geography_str, time_range)

    verified_events = len(
        [e for e in final_events if e['validation_status'] == 'verified']
    )
    terminal_status = 'accepted' if verified_events > 0 else 'unverified_only'
    set_terminal_state(
        config,
        terminal_status,
        'pipeline_completed',
        current_stage,
        total_events=len(final_events),
        verified_events=verified_events,
        safe_to_accept=verified_events > 0,
        workflow_outcome=terminal_status,
    )
    persist_workflow_outcome(
        config,
        terminal_status,
        total_events=len(final_events),
        verified_events=verified_events,
    )

    final_dataset = events_to_dataframe(final_events)

    logger.info(f'Final events consolidated: {len(final_events)}')

    return final_dataset

  except ProviderSemanticError as e:
    set_terminal_state(
        config,
        e.terminal_status,
        str(e),
        current_stage,
        total_events=0,
        verified_events=0,
        safe_to_accept=False,
        workflow_outcome=e.terminal_status,
    )
    persist_workflow_outcome(config, e.terminal_status)
    logger.warning(f'Provider semantic termination reached: {str(e)}')
    return None
  except Exception as e:
    set_terminal_state(
        config,
        'failed',
        f'pipeline_exception: {str(e)}',
        current_stage,
        total_events=0,
        verified_events=0,
        safe_to_accept=False,
        workflow_outcome='failed',
    )
    persist_workflow_outcome(config, 'failed')
    logger.error(
        "Pipeline failed during stage '%s': %s",
        current_stage,
        str(e),
        exc_info=True,
    )
    return None


async def parallel_discovery_stage(
    keyword,
    time_range,
    domain,
    geography_str,
    geography,
    config,
):
  """Executes parallel LLM event discovery and trend-based peak discovery.

  Args:
    keyword: Target search variable or keyword.
    time_range: Dictionary specifying start and end time range.
    domain: Domain of analysis.
    geography_str: String representation of geography.
    geography: Dictionary containing geography configuration.
    config: Pipeline configuration dictionary.

  Returns:
    Tuple containing initial LLM events list and significant peaks list.
  """
  logger.info('Running parallel discovery stage.')

  dual_channel_config = config.get('dual_channel_discovery', {})
  llm_enabled = dual_channel_config.get('llm_channel', {}).get('enabled', True)
  trends_enabled = dual_channel_config.get('trends_channel', {}).get(
      'enabled', True
  )

  significant_peaks = []
  if trends_enabled:
    logger.info('Starting trend-based peak discovery.')
    try:
      significant_peaks = get_significant_peaks_via_trends(
          keyword, time_range, geography, config
      )
      logger.info(f'Significant peaks discovered: {len(significant_peaks)}')
    except Exception as e:
      logger.warning(f'Parallel discovery failed: {str(e)}')
      significant_peaks = []

  initial_events = []
  if llm_enabled:
    logger.info('Starting LLM-based event discovery.')
    initial_events = await llm_discovery_with_coverage_supervision(
        keyword, time_range, domain, geography_str, config, significant_peaks
    )
    logger.info(
        f'LLM-based discovery returned {len(initial_events)} initial events.'
    )

  if initial_events and significant_peaks:
    final_coverage = calculate_coverage_rate(
        initial_events, significant_peaks, config
    )
    logger.info(f'Final coverage rate: {final_coverage:.1%}')
  elif significant_peaks:
    logger.info(f'Significant peaks detected: {len(significant_peaks)}')
  elif initial_events:
    logger.info(f'Initial events detected: {len(initial_events)}')

  logger.info(f'Channel A event count: {len(initial_events)}')
  logger.info(f'Channel B peak count: {len(significant_peaks)}')

  return initial_events, significant_peaks


def cross_validation_and_allocation_stage(
    initial_events, significant_peaks, config
):
  """Cross-validates initial events with search peaks and allocates tasks.

  Args:
    initial_events: List of initial events discovered by LLM.
    significant_peaks: List of significant search peaks discovered via trends.
    config: Pipeline configuration dictionary.

  Returns:
    Tuple containing allocated validation tasks list and exploration tasks list.
  """
  logger.info('Starting cross-validation and task allocation.')

  validation_tasks, exploration_tasks = cross_validate_and_allocate(
      initial_events, significant_peaks, config
  )

  task_summary = generate_task_summary(validation_tasks, exploration_tasks)

  logger.info('Task allocation summary:')
  logger.info(f"  Total tasks: {task_summary['total_tasks']}")
  logger.info(
      f"  Validation tasks: {task_summary['validation_tasks']['count']}"
  )
  logger.info(
      f"  Exploration tasks: {task_summary['exploration_tasks']['count']}"
  )
  logger.info(f"  Total query budget: {task_summary['total_query_budget']}")

  return validation_tasks, exploration_tasks


async def execution_and_consolidation_stage(
    validation_tasks,
    exploration_tasks,
    keyword,
    config,
):
  """Executes validation and exploration tasks, consolidating all final events.

  Args:
    validation_tasks: List of validation tasks to execute.
    exploration_tasks: List of exploration tasks to execute.
    keyword: Target search variable or keyword.
    config: Pipeline configuration dictionary.

  Returns:
    List of consolidated final event dictionaries.
  """
  logger.info('Starting execution and consolidation.')

  final_events = []

  if validation_tasks:
    logger.info(f'Executing {len(validation_tasks)} validation tasks.')
    validation_results = await execute_validation_tasks(
        validation_tasks, keyword, config
    )
    final_events.extend(validation_results)

  run_exploration = config.get('event_sourcing', {}).get(
      'run_trend_exploration_after_validation', True
  )

  if exploration_tasks and run_exploration:
    logger.info(f'Executing {len(exploration_tasks)} exploration tasks.')
    exploration_results = await execute_exploration_tasks(
        exploration_tasks, keyword, config
    )
    final_events.extend(exploration_results)
  elif exploration_tasks and not run_exploration:
    logger.info(
        f'Skipping {len(exploration_tasks)} exploration tasks because '
        'run_trend_exploration_after_validation=false.'
    )
  elif not exploration_tasks:
    logger.info('Stage transition complete.')

  logger.info(f'Final events consolidated: {len(final_events)}')

  return final_events


async def execute_validation_tasks(
    validation_tasks, keyword, config
):
  """Executes validation tasks iteratively and appends results to dataset.

  Args:
    validation_tasks: List of validation task dictionaries.
    keyword: Target search variable or keyword.
    config: Pipeline configuration dictionary.

  Returns:
    List of validated event dictionaries.
  """
  logger.info('Running validation task loop.')

  validated_events = []

  for i, task in enumerate(validation_tasks, 1):
    logger.info(
        f"Validation task {i}/{len(validation_tasks)}: {task['task_id']}"
    )

    try:

      researcher = IterativeResearcher(config, 'validation', task)

      final_summary, research_tree = (
          await researcher.investigate_validation_task()
      )

      validated_event = integrate_validation_result(
          task, final_summary, research_tree
      )
      validated_events.append(validated_event)

      append_event_to_dataset(validated_event, config)

      await save_research_tree(research_tree, task['task_id'], config)

      logger.info(f"Completed validation task {task['task_id']}.")

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"Validation task {task['task_id']} failed: {str(e)}")

      failed_event = create_failed_event_record(task, str(e))
      validated_events.append(failed_event)

      append_event_to_dataset(failed_event, config)
      continue

  logger.info(f'Validated events generated: {len(validated_events)}')

  return validated_events


async def execute_exploration_tasks(
    exploration_tasks, keyword, config
):
  """Executes exploration tasks iteratively and appends results to dataset.

  Args:
    exploration_tasks: List of exploration task dictionaries.
    keyword: Target search variable or keyword.
    config: Pipeline configuration dictionary.

  Returns:
    List of explored event dictionaries.
  """
  logger.info('Running exploration task loop.')

  discovered_events = []

  for i, task in enumerate(exploration_tasks, 1):
    logger.info(
        f"Exploration task {i}/{len(exploration_tasks)}: {task['task_id']}"
    )

    try:

      researcher = IterativeResearcher(config, 'exploration', task)

      final_summary, research_tree = (
          await researcher.investigate_exploration_task()
      )

      discovered_event = integrate_exploration_result(
          task, final_summary, research_tree
      )
      discovered_events.append(discovered_event)

      append_event_to_dataset(discovered_event, config)

      await save_research_tree(research_tree, task['task_id'], config)

      logger.info(f"Completed exploration task {task['task_id']}.")

    except ProviderSemanticError:
      raise
    except Exception as e:
      logger.error(f"Exploration task {task['task_id']} failed: {str(e)}")

      failed_event = create_failed_event_record(task, str(e))
      discovered_events.append(failed_event)

      append_event_to_dataset(failed_event, config)
      continue

  logger.info(f'Exploration events generated: {len(discovered_events)}')

  return discovered_events


def integrate_validation_result(
    task, final_summary, research_tree
):
  """Integrates a validation task result into the final dataset schema.

  Args:
    task: Validation task dictionary.
    final_summary: Final summary dictionary produced by the researcher.
    research_tree: Research tree dictionary containing investigation history.

  Returns:
    Integrated event dictionary formatted to dataset schema.
  """
  event_data = task['event_data']
  matched_peaks = task.get('matched_peaks', [])
  trend_correlation = calculate_trend_correlation(matched_peaks)
  verification_path = analyze_verification_path(research_tree)

  validated_urls_A = []
  validated_urls_B = []
  source_urls_initial = []
  source_urls_followup = []
  source_urls_all_crawled = []
  source_status_counts = {}

  def visit_node(node):
    node_query = str(node.get('query', ''))
    is_initial = node_query.startswith('Direct URL processing:')
    for source in node.get('sources', []) or []:
      url = source.get('url')
      if not url:
        continue
      status = source.get('status') or 'unknown'
      source_urls_all_crawled.append(url)
      source_status_counts[status] = source_status_counts.get(status, 0) + 1
      if is_initial:
        source_urls_initial.append(url)
      else:
        source_urls_followup.append(url)
      if status == 'success':
        validated_urls_A.append(url)
      elif source.get('cleaning_status') == 'success' or status == 'cleaned':
        validated_urls_B.append(url)
    for child in node.get('children', []) or []:
      visit_node(child)

  if research_tree:
    visit_node(research_tree)

  integrated_event = {
      'event_id': event_data['event_id'],
      'discovery_channel': 'llm',
      'event_summary_validated': final_summary.get(
          'summary', event_data['event_summary']
      ),
      'timing': {
          'announcement_date': (
              final_summary.get('timing_info', {}).get('announcement_date')
              or event_data.get('announcement_date')
          ),
          'occurrence_date': (
              final_summary.get('timing_info', {}).get('occurrence_date')
              or event_data.get('occurrence_date')
          ),
          'type': event_data.get('event_type', 'Instantaneous'),
      },
      'validation_status': determine_validation_status(final_summary),
      'verification_path': verification_path,
      'trend_correlation': trend_correlation,
      'source_urls_validated_A': sorted(set(validated_urls_A)),
      'source_urls_validated_B': sorted(set(validated_urls_B)),
      'source_urls_initial': sorted(set(source_urls_initial)),
      'source_urls_followup': sorted(set(source_urls_followup)),
      'source_urls_all_crawled': sorted(set(source_urls_all_crawled)),
      'source_url_quality_summary': {
          'total_crawled': len(set(source_urls_all_crawled)),
          'status_counts': source_status_counts,
      },
      'original_event_claim': {
          'event_summary': event_data.get('event_summary'),
          'announcement_date': event_data.get('announcement_date'),
          'occurrence_date': event_data.get('occurrence_date'),
          'event_type': event_data.get('event_type'),
          'source_urls': event_data.get('source_urls', []),
      },
      'research_tree_file': f"validation_task_{event_data['event_id']}.json",
      'task_metadata': {
          'task_id': task['task_id'],
          'query_budget_used': research_tree.get('total_queries_executed', 0),
          'initial_source_urls_processed': research_tree.get(
              'initial_source_urls_processed', 0
          ),
          'followup_search_queries_executed': research_tree.get(
              'followup_search_queries_executed', 0
          ),
          'processing_time': research_tree.get('processing_time_minutes', 0),
      },
      'llm_corrected_data': {
          'reasoning_for_date_choice': final_summary.get(
              'synthesis_reasoning', 'No reasoning provided'
          ),
          'supporting_quotes': final_summary.get('supporting_quotes', []),
          'evidence_statistics': {
              'confirmed_statements_count': final_summary.get(
                  'confirmed_statements_count', 0
              ),
              'anticipated_statements_count': final_summary.get(
                  'anticipated_statements_count', 0
              ),
              'internal_resolutions_count': final_summary.get(
                  'internal_resolutions_count', 0
              ),
          },
      },
      'original_timing': {
          'announcement_date': event_data.get('announcement_date'),
          'occurrence_date': event_data.get('occurrence_date'),
      },
  }

  return integrated_event


def integrate_exploration_result(
    task, final_summary, research_tree
):
  """Integrates an exploration task result into the final dataset schema.

  Args:
    task: Exploration task dictionary.
    final_summary: Final summary dictionary produced by the researcher.
    research_tree: Research tree dictionary containing investigation history.

  Returns:
    Integrated event dictionary formatted to dataset schema.
  """
  peak_data = task['peak_data']

  event_summary = final_summary.get(
      'summary',
      f"Unknown event causing search spike on {peak_data['peak_date'].date()}",
  )

  verification_path = analyze_verification_path(research_tree)

  integrated_event = {
      'event_id': f"exploration_{peak_data['peak_date'].strftime('%Y%m%d')}",
      'discovery_channel': 'trends',
      'event_summary_validated': event_summary,
      'timing': {
          'announcement_date': peak_data['peak_date'].strftime('%Y-%m-%d'),
          'occurrence_date': peak_data['peak_date'].strftime('%Y-%m-%d'),
          'type': 'Instantaneous',
      },
      'validation_status': determine_validation_status(final_summary),
      'verification_path': verification_path,
      'trend_correlation': {
          'peak_date': peak_data['peak_date'].strftime('%Y-%m-%d'),
          'peak_value': peak_data['peak_value'],
          'correlation_score': 1.0,
      },
      'source_urls_validated': research_tree.get('source_urls', []),
      'research_tree_file': (
          f"exploration_task_{peak_data['peak_date'].strftime('%Y%m%d')}.json"
      ),
      'task_metadata': {
          'task_id': task['task_id'],
          'query_budget_used': research_tree.get('total_queries_executed', 0),
          'processing_time': research_tree.get('processing_time_minutes', 0),
      },
  }

  return integrated_event


def calculate_trend_correlation(matched_peaks):
  """Calculates correlation metrics for the highest peak among matched peaks.

  Args:
    matched_peaks: List of matched peak dictionaries.

  Returns:
    Dictionary containing peak date, peak value, and correlation score.
  """
  if not matched_peaks:
    return {'peak_date': None, 'peak_value': 0, 'correlation_score': 0.0}

  best_peak = max(matched_peaks, key=lambda x: x['peak_value'])

  return {
      'peak_date': best_peak['peak_date'].strftime('%Y-%m-%d'),
      'peak_value': best_peak['peak_value'],
      'correlation_score': min(best_peak['peak_value'] / 100.0, 1.0),
  }


def determine_validation_status(final_summary):
  """Determines the standardized validation status from the final summary.

  Args:
    final_summary: Final summary dictionary containing category and text.

  Returns:
    Standardized validation status string.
  """
  summary_text = final_summary.get('summary', '').lower()
  category = final_summary.get('category', 'None')

  logger.debug(f"Status diagnostics: summary category: '{category}'")
  logger.debug(f"Status diagnostics: summary text100: '{summary_text[:100]}'")

  if category == 'verified':
    result = 'verified'
  elif category == 'Error' or 'error' in summary_text:
    result = 'error'
  elif category == 'None' or 'no relevant event' in summary_text:
    result = 'no_event_found'
  elif 'conflicting' in summary_text or 'uncertain' in summary_text:
    result = 'conflicting_reports'
  elif category == 'unverified' or 'could not be verified' in summary_text:
    result = 'unverified'
  else:

    result = 'unverified'

  logger.info(
      f"Status diagnostics: category='{category}' ->"
      f" validation_status='{result}'"
  )
  return result


def create_failed_event_record(task, error_message):
  """Creates a fallback event record representing a failed task execution.

  Args:
    task: Task dictionary (validation or exploration).
    error_message: Error message string explanation.

  Returns:
    Event dictionary indicating execution failure.
  """
  if task['task_type'] == 'validation':
    event_data = task['event_data']
    event_id = event_data['event_id']
  else:
    peak_data = task['peak_data']
    event_id = f"exploration_{peak_data['peak_date'].strftime('%Y%m%d')}"

  return {
      'event_id': event_id,
      'discovery_channel': (
          'llm' if task['task_type'] == 'validation' else 'trends'
      ),
      'event_summary_validated': f'Task execution failed: {error_message}',
      'timing': {
          'announcement_date': None,
          'occurrence_date': None,
          'type': 'Unknown',
      },
      'validation_status': 'execution_failed',
      'verification_path': 'failed',
      'trend_correlation': {
          'peak_date': None,
          'peak_value': 0,
          'correlation_score': 0.0,
      },
      'source_urls_validated': [],
      'research_tree_file': None,
      'task_metadata': {'task_id': task['task_id'], 'error': error_message},
  }


async def save_research_tree(research_tree, task_id, config):
  """Saves the research tree JSON data to the configured output directory.

  Args:
    research_tree: Research tree dictionary to save.
    task_id: Identifier for the task.
    config: Pipeline configuration dictionary.
  """
  try:

    research_output_dir = config['paths'].get(
        'research_output_dir', 'data/research_trees'
    )
    os.makedirs(research_output_dir, exist_ok=True)

    filename = f'{task_id}.json'
    filepath = os.path.join(research_output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
      json.dump(research_tree, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f'Saved research tree JSON file: {filepath}')

  except Exception as e:
    logger.warning(
        'Failed to save research tree for task %s: %s', task_id, str(e)
    )


def events_to_dataframe(events):
  """Flattens event dictionaries and converts them into a pandas DataFrame.

  Args:
    events: List of event dictionaries.

  Returns:
    Pandas DataFrame containing flattened event data with parsed datetimes.
  """
  if not events:
    return pd.DataFrame()

  flattened_data = []

  for event in events:
    flat_event = {
        'event_id': event['event_id'],
        'discovery_channel': event['discovery_channel'],
        'event_summary': event['event_summary_validated'],
        'announcement_date': event['timing']['announcement_date'],
        'occurrence_date': event['timing']['occurrence_date'],
        'event_type': event['timing']['type'],
        'validation_status': event['validation_status'],
        'peak_date': event['trend_correlation']['peak_date'],
        'peak_value': event['trend_correlation']['peak_value'],
        'correlation_score': event['trend_correlation']['correlation_score'],
        'research_tree_file': event['research_tree_file'],
    }
    flattened_data.append(flat_event)

  df = pd.DataFrame(flattened_data)

  date_columns = ['announcement_date', 'occurrence_date', 'peak_date']
  for col in date_columns:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')

  return df


def analyze_verification_path(research_tree):
  """Analyzes the investigation tree to determine the verification path taken.

  Args:
    research_tree: Research tree dictionary containing investigation execution.

  Returns:
    String representing verification path ('first_round' or
    'supplemental_search').
  """
  if not research_tree or 'children' not in research_tree:
    return 'first_round'

  action_plan_executed = False

  for child in research_tree['children']:

    if child.get('node_type') == 'query' or not child.get('is_initial', True):
      action_plan_executed = True
      break

    if 'sources' in child:
      for source in child['sources']:
        if not source.get('is_initial', True):
          action_plan_executed = True
          break

  return 'supplemental_search' if action_plan_executed else 'first_round'


def calculate_verification_breakdown(events):
  """Calculates the breakdown of verification statuses and paths across events.

  Args:
    events: List of event dictionaries.

  Returns:
    Dictionary containing counts for each verification status and path category.
  """
  breakdown = {
      'first_round_verified': 0,
      'supplemental_search_verified': 0,
      'rejected_unverified': 0,
      'rejected_no_event': 0,
      'rejected_conflicting': 0,
      'execution_failed': 0,
  }

  for event in events:
    status = event.get('validation_status', 'unknown')
    path = event.get('verification_path', 'unknown')

    if status == 'verified':
      if path == 'first_round':
        breakdown['first_round_verified'] += 1
      else:
        breakdown['supplemental_search_verified'] += 1
    elif status == 'unverified':
      breakdown['rejected_unverified'] += 1
    elif status == 'no_event_found':
      breakdown['rejected_no_event'] += 1
    elif status == 'conflicting_reports':
      breakdown['rejected_conflicting'] += 1
    elif status == 'execution_failed':
      breakdown['execution_failed'] += 1

  return breakdown


def calculate_url_quality_stats(events, config):
  """Calculates URL quality statistics and invalid URL ratios from research trees.

  Args:
    events: List of event dictionaries.
    config: Pipeline configuration dictionary.

  Returns:
    Dictionary containing URL crawling statistics and quality metrics.
  """
  total_initial_urls = 0
  successful_urls = 0
  failed_urls = 0
  empty_urls = 0

  for event in events:

    research_tree_file = event.get('research_tree_file')
    if research_tree_file:
      try:
        research_output_dir = config.get('paths', {}).get(
            'research_output_dir', 'data/research_trees'
        )
        tree_path = os.path.join(research_output_dir, research_tree_file)

        if os.path.exists(tree_path):
          with open(tree_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

          if 'children' in tree_data:
            for child in tree_data['children']:
              if 'sources' in child:
                for source in child['sources']:

                  if source.get('is_initial', True):
                    total_initial_urls += 1
                    status = source.get('status', 'unknown')
                    if status == 'success':
                      successful_urls += 1
                    elif status == 'failed':
                      failed_urls += 1
                    elif status == 'empty':
                      empty_urls += 1
      except Exception as e:
        logger.warning(
            'Failed to inspect research tree %s: %s', research_tree_file, str(e)
        )

  invalid_url_ratio = (
      (failed_urls + empty_urls) / total_initial_urls
      if total_initial_urls > 0
      else 0.0
  )

  return {
      'total_initial_urls': total_initial_urls,
      'successful_urls': successful_urls,
      'failed_urls': failed_urls,
      'empty_urls': empty_urls,
      'invalid_url_ratio': round(invalid_url_ratio, 3),
  }
