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

"""Module 01: Google Trends Data Fetching"""

import logging
import os
import re
import time
from typing import Dict, List
import pandas as pd
from pytrends.request import TrendReq
import urllib3
from urllib3.exceptions import InsecureRequestWarning

logger = logging.getLogger(__name__)


def clean_column_name(name):
  """Clean column names to be descriptive and valid

  Args:
      name: Original column name

  Returns:
      Cleaned column name
  """
  # Remove special characters and convert to lowercase
  cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", name.lower())
  # Replace spaces with underscores
  cleaned = re.sub(r"\s+", "_", cleaned)
  return cleaned


def create_robust_pytrends():
  """Create a robust TrendReq instance with optional proxy and retry mechanisms.

  Proxy credentials are loaded from environment variables.
  If proxy credentials are not configured, this function falls back to a
  retry-enabled TrendReq instance without proxy.

  Returns:
      TrendReq instance configured with retries and optional proxy.
  """
  # Suppress SSL warnings for proxy usage
  urllib3.disable_warnings(InsecureRequestWarning)

  username_base = os.getenv("OXYLABS_USERNAME_BASE", "").strip()
  password = os.getenv("OXYLABS_PASSWORD", "").strip()
  proxy_host = os.getenv("OXYLABS_PROXY_HOST", "").strip()
  proxy_country = os.getenv("OXYLABS_PROXY_COUNTRY", "US").strip() or "US"
  proxy_port_raw = os.getenv("OXYLABS_PROXY_PORT", "").strip()

  proxy_configured = bool(
      username_base and password and proxy_host and proxy_country
  )
  proxy_port = 8000
  if proxy_port_raw:
    try:
      proxy_port = int(proxy_port_raw)
    except ValueError:
      logger.warning(
          "Invalid OXYLABS_PROXY_PORT: %s. Falling back to default port %s.",
          proxy_port_raw,
          proxy_port,
      )

  kwargs = {
      "hl": "en-US",
      "tz": 360,
      "timeout": (10, 25),
      "retries": 3,
      "backoff_factor": 0.2,
      "requests_args": {"verify": False},
  }

  if proxy_configured:
    formatted_username = f"user-{username_base}-country-{proxy_country}"
    proxy_url = (
        f"https://{formatted_username}:{password}@{proxy_host}:{proxy_port}"
    )
    logger.info("Switching to robust mode with proxy and retry mechanisms")
    logger.info(
        f"Using proxy: {proxy_host}:{proxy_port} with country: {proxy_country}"
    )
    kwargs["proxies"] = [proxy_url]
  else:
    missing = [
        name
        for name, value in {
            "OXYLABS_USERNAME_BASE": username_base,
            "OXYLABS_PASSWORD": password,
            "OXYLABS_PROXY_HOST": proxy_host,
            "OXYLABS_PROXY_COUNTRY": proxy_country,
        }.items()
        if not value
    ]
    if missing:
      logger.warning(
          "Oxylabs proxy credentials are incomplete or missing: %s. "
          "Falling back to retry-enabled request without proxy.",
          ", ".join(missing),
      )
    else:
      logger.warning(
          "Oxylabs proxy is disabled. Using retry-enabled request without"
          " proxy."
      )

  return TrendReq(**kwargs)


def is_rate_limit_error(error_message):
  """Check if an error message indicates a rate limit (429) error

  Args:
      error_message: The error message to check

  Returns:
      True if this appears to be a rate limit error
  """
  rate_limit_indicators = [
      "code 429",
      "Too Many Requests",
      "rate limit",
      "quota exceeded",
  ]

  error_lower = str(error_message).lower()
  return any(
      indicator.lower() in error_lower for indicator in rate_limit_indicators
  )


def fetch_google_trends(keywords, config):
  """Fetch data from Google Trends for specified keywords with progressive enhancement:

  - Start with normal requests (no proxy)
  - Switch to proxy+retry mode if rate limit (429) errors are encountered
  Each keyword is requested independently to avoid cross-keyword normalization

  Args:
      keywords: List of keywords to fetch trends for
      config: Configuration dictionary

  Returns:
      DataFrame with independent trends data for each keyword
  """
  logger.info(
      f"Fetching Google Trends data for {len(keywords)} keywords independently"
  )
  logger.info(f"Keywords: {keywords}")

  try:
    # Initialize with standard pytrends (no proxy initially)
    pytrends = TrendReq(hl="en-US", tz=360)
    using_robust_mode = False

    # Prepare timeframe
    start_date = config["time_range"]["start"]
    end_date = config["time_range"]["end"]
    timeframe = f"{start_date} {end_date}"

    # Construct geo parameter from geography configuration
    geography = config["geography"]
    country_code = geography["country_code"]
    subdivision_code = geography["subdivision_code"]

    if subdivision_code == "World":
      geo_string = ""
      logger.info("Using worldwide scope")
    else:
      geo_string = country_code
      logger.info(f"Using geographic scope: {country_code}")

    # List to store individual trend DataFrames
    all_trends_dfs = []

    # Process each keyword independently
    for i, keyword in enumerate(keywords):
      logger.info(f"Processing keyword {i+1}/{len(keywords)}: '{keyword}'")

      try:
        # Build payload for single keyword
        pytrends.build_payload(
            kw_list=[keyword],  # Single keyword only
            cat=0,
            timeframe=timeframe,
            geo=geo_string,
            gprop="",
        )

        # Add delay to respect rate limits
        time.sleep(1)

        # Get interest over time data for this keyword
        keyword_data = pytrends.interest_over_time()

        if not keyword_data.empty:
          # Drop 'isPartial' column if it exists
          if "isPartial" in keyword_data.columns:
            keyword_data = keyword_data.drop("isPartial", axis=1)

          # Rename the column appropriately
          if i == 0:  # First keyword is the target variable
            keyword_data = keyword_data.rename(columns={keyword: "target"})
            logger.info(f"Fetched target variable '{keyword}' successfully.")
          else:  # Rest are specific covariates
            clean_name = f"cov_specific_{clean_column_name(keyword)}"
            keyword_data = keyword_data.rename(columns={keyword: clean_name})
            logger.info(f"Fetched covariate '{keyword}' as '{clean_name}'.")

          # Add to collection
          all_trends_dfs.append(keyword_data)
        else:
          logger.warning(f"No data returned for keyword '{keyword}'.")

      except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Failed to fetch data for keyword '{keyword}': {error_msg}"
        )

        # Check if this is a rate limit error and we haven't switched to robust mode yet
        if is_rate_limit_error(error_msg) and not using_robust_mode:
          logger.warning(
              "Rate limit detected! Switching to robust mode with proxy and"
              " retries..."
          )

          try:
            # Switch to robust pytrends instance
            pytrends = create_robust_pytrends()
            using_robust_mode = True

            # Wait a bit before retrying
            logger.info("Waiting 5 seconds before retrying with robust mode...")
            time.sleep(5)

            # Retry the failed keyword with robust instance
            logger.info(f"Retrying keyword '{keyword}' with robust mode...")
            pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=timeframe,
                geo=geo_string,
                gprop="",
            )

            # Additional delay for robust mode
            time.sleep(2)

            keyword_data = pytrends.interest_over_time()

            if not keyword_data.empty:
              if "isPartial" in keyword_data.columns:
                keyword_data = keyword_data.drop("isPartial", axis=1)

              if i == 0:
                keyword_data = keyword_data.rename(columns={keyword: "target"})
                logger.info(
                    f"Fetched target variable '{keyword}' successfully using"
                    " robust mode."
                )
              else:
                clean_name = f"cov_specific_{clean_column_name(keyword)}"
                keyword_data = keyword_data.rename(
                    columns={keyword: clean_name}
                )
                logger.info(
                    f"Fetched covariate '{keyword}' as '{clean_name}' using"
                    " robust mode."
                )

              all_trends_dfs.append(keyword_data)
            else:
              logger.warning(
                  f"No data returned for keyword '{keyword}' in robust mode."
              )

          except Exception as retry_error:
            logger.error(
                f"Failed to fetch data for keyword '{keyword}' even in robust"
                f" mode: {str(retry_error)}"
            )
            continue
        else:
          # Either not a rate limit error or already in robust mode
          continue

    # Check if we have any successful data
    if not all_trends_dfs:
      logger.error("No trends data was successfully fetched for any keyword")
      return pd.DataFrame()

    # Merge all individual DataFrames
    logger.info(f"Merging {len(all_trends_dfs)} independent trend datasets")
    trends_data = pd.concat(all_trends_dfs, axis=1)

    # Ensure index is datetime
    trends_data.index = pd.to_datetime(trends_data.index)
    trends_data.index.name = "timestamp"

    logger.info(
        "Successfully fetched independent trends data with shape:"
        f" {trends_data.shape}"
    )
    logger.info(
        f"Date range: {trends_data.index.min()} to {trends_data.index.max()}"
    )
    logger.info(f"Columns: {list(trends_data.columns)}")

    if using_robust_mode:
      logger.info(
          "Completed data fetching using robust mode (proxy + retries)."
      )
    else:
      logger.info("Completed data fetching using standard mode.")

    return trends_data

  except Exception as e:
    logger.error(f"Error fetching Google Trends data: {str(e)}")
    raise
