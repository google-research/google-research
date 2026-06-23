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

"""Module 00: Project Setup and Configuration Validation"""

from datetime import datetime
import logging
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, validator
import yaml


class LLMModels(BaseModel):
  initial_event_acquisition: Optional[str] = "gemini-3.5-flash"
  single_source_summary: Optional[str] = "gemini-3.5-flash"
  information_sufficiency_evaluator: Optional[str] = "gemini-3.5-flash"
  claim_verifier: Optional[str] = "gemini-3.5-flash"
  final_synthesis: Optional[str] = "gemini-3.5-flash"


class TimeRange(BaseModel):
  start: str
  end: str

  @validator("start", "end")
  def validate_date_format(cls, v):
    try:
      datetime.strptime(v, "%Y-%m-%d")
      return v
    except ValueError:
      raise ValueError("Date must be in YYYY-MM-DD format")


class Geography(BaseModel):
  country_code: str
  subdivision_code: Optional[str] = None
  city: Optional[str] = None


class RecalibrationConfig(BaseModel):
  enabled: bool = False
  recalibration_model: str = "gemini-3.5-flash"


class EventSourcing(BaseModel):
  iterative_crawl_count: int
  save_research_trees: bool
  max_content_length: int = 15000
  reconnaissance_query_templates: List[str]
  run_trend_exploration_after_validation: bool = True
  recalibration: Optional[RecalibrationConfig] = None
  initial_event_iterations: int = 1
  max_initial_events: int = 100
  max_evaluation_actions: int = 5
  coverage_early_stop_enabled: bool = False
  coverage_early_stop_threshold: float = 0.95
  coverage_min_acceptable: float = 0.90
  coverage_time_window_days: int = 5


class Paths(BaseModel):
  output_root: Optional[str] = None
  raw_trends: str
  raw_weather: str
  raw_nasdaq: str
  final_dataset: str
  log_file: Optional[str] = None
  research_output_dir: Optional[str] = None


class Config(BaseModel):
  llm_models: LLMModels
  domain: Optional[str] = "General Analysis"
  keywords_to_process: Optional[List[str]] = []
  processing_mode: Optional[Dict] = {
      "mode": "full",
      "status_file": "data/processed_keywords.log",
  }
  project_name: str
  geography_str: Optional[str] = "United States"
  geography: Geography
  time_range: TimeRange
  frequency: str
  target_variable: Optional[str] = None
  event_sourcing: EventSourcing
  dual_channel_discovery: Optional[Dict] = {}
  cross_validation: Optional[Dict] = {}
  task_execution: Optional[Dict] = {}
  paths: Paths


def load_and_validate_config(config_path):
  """Load and validate configuration from YAML file

  Args:
      config_path: Path to the config.yaml file

  Returns:
      Validated configuration dictionary

  Raises:
      FileNotFoundError: If config file doesn't exist
      ValidationError: If config doesn't match schema
  """
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

  with open(config_path, "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

  # Validate using Pydantic model
  validated_config = Config(**config_data)

  # Return as dictionary for easier use
  return validated_config.dict()


def setup_project(config):
  """Setup project directories and configure logging

  Args:
      config: Validated configuration dictionary
  """
  # Create directories
  directories = ["data/raw", "data/processed", "data/final", "logs"]

  for directory in directories:
    os.makedirs(directory, exist_ok=True)

  # Clear existing handlers to prevent duplicate logging
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

  # Configure logging
  log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # Get log file path from config or use default
  log_file_path = config["paths"].get("log_file", "logs/chronicle_weaver.log")

  # Ensure log directory exists
  os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

  logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
  )

  logger = logging.getLogger(__name__)
  logger.info(f"Project setup complete for: {config['project_name']}")
  logger.info(f"Target variable: {config['target_variable']}")
  logger.info(
      f"Time range: {config['time_range']['start']} to"
      f" {config['time_range']['end']}"
  )

  # Log geography information
  geography = config["geography"]
  geo_parts = [geography["country_code"]]
  if geography.get("subdivision_code"):
    geo_parts.append(geography["subdivision_code"])
  if geography.get("city"):
    geo_parts.append(geography["city"])
  logger.info(f"Geographic scope: {', '.join(geo_parts)}")
