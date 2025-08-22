# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Convenience conversion methods between different time representations.

In all methods there is an explicit requirement to state the timezone.
There are convenience methods for UTC and Japan timezones, because these are the
two most commonly used in our codebase.
"""

import datetime
from dateutil import tz

import pytz

JAPAN_TIMEZONE = pytz.timezone('Japan')
PS_TIMEZONE = pytz.timezone('US/Pacific')
UTC_TIMEZONE = pytz.timezone('UTC')


def time_to_bytes_utc(timestamp):
  """Encodes time in seconds since Epoch as a byte-string of ISO8601 format."""
  return datetime.datetime.utcfromtimestamp(timestamp).isoformat().encode()


def time_to_datetime(
    timestamp, timezone
):
  """Converts time in seconds since Epoch to datetime.

  Args:
    timestamp: Seconds since Epoch time.
    timezone: The timezone of the output datetime.

  Returns:
    A datetime representation.
  """
  return datetime.datetime.fromtimestamp(timestamp, tz=timezone)


def time_to_datetime_utc(timestamp):
  """Converts time in seconds since Epoch to datetime in UTC timezone."""
  return time_to_datetime(timestamp, UTC_TIMEZONE)


def time_to_datetime_japan(timestamp):
  """Converts time in seconds since Epoch to datetime in Japan timezone."""
  return time_to_datetime(timestamp, JAPAN_TIMEZONE)


def time_to_datetime_pst(timestamp):
  """Converts time in seconds since Epoch to datetime in PST."""
  return time_to_datetime(timestamp, PS_TIMEZONE)


def datetime_to_time(
    timezone,
    year,
    month = 1,
    day = 1,
    hour = 0,
    minute = 0,
    second = 0,
    microsecond = 0,
):
  """Converts a date and time to seconds since Epoch.

  Although this can be achieved with datetime.datetime, this method (and the UTC
  and Japan versions below) are convenient, and used throughout our code,
  especially in tests.
  The default values allow to just state the year (or year and month, or day,
  etc.), and get the timestamp for the start of that year (or that month, that
  day, etc.).

  Args:
    timezone: The timezone of the input date.
    year: The year of the input date.
    month: The month of the input date.
    day: The day of the input date.
    hour: The hour of the input time.
    minute: The minute of the input time.
    second: The second of the input time.
    microsecond: The microsecond of the input time.

  Returns:
    A timestamp, in seconds since Epoch.
  """
  return int(
      _create_datetime(
          year=year,
          month=month,
          day=day,
          hour=hour,
          minute=minute,
          second=second,
          microsecond=microsecond,
          tzinfo=timezone,
      ).timestamp()
  )


def _create_datetime(
    year,
    month,
    day,
    hour=0,
    minute=0,
    second=0,
    microsecond=0,
    tzinfo=None,
    is_dst=False,
    fold=0
    ):
  """Wrapper function for the standard datetime.datetime constructor.

  pytz timezone object require different style of initialization that
  caused a lot of confusion.  This function works for both pytz and
  non-pytz timezone objects.

  Args:
    year: datetime.MINYEAR <= year <= datetime.MAXYEAR
    month: 1 <= month <= 12
    day: 1 <= day <= number of days in the given month and year
    hour: 0 <= hour < 24
    minute: 0 <= minute < 60
    second: 0 <= second < 60
    microsecond: 0 <= microsecond < 1000000
    tzinfo: Timezone info, can be pytz timezone or something else.
    is_dst: Whether the time is in daylight saving or not. Used only for pytz
      and effective only for ambiguous time at the border of daylight saving
      period.
    fold: [0, 1] - during an ambiguous or imaginary time, this is used to decide
      whether the UTC offset used should be the value before the transition (0)
      or after (1).

  Returns:
    A new datetime object, local to the specified timezone.
    Without tzinfo, a naive datetime object is returned.

  Raises:
    ValueError for the values out of range.
    pytz.exceptions.AmbiguousTimeError if a pytz zone is used with is_dst=None
    and the time is ambiguous.
    pytz.exceptions.NonExistentTimeError if a pytz zone is used withis_dst=None
    and the time is non-existent.
  """

  if hasattr(tzinfo, 'localize'):
    return tzinfo.localize(datetime.datetime(year, month, day, hour, minute,
                                             second, microsecond),
                           is_dst=is_dst)
  else:
    # pylint: disable=g-tzinfo-datetime
    kwargs = {'fold': fold}

    dt = datetime.datetime(year, month, day, hour, minute, second, microsecond,
                           tzinfo, **kwargs)

    if not kwargs:
      dt = tz.enfold(dt, fold=fold)

    return dt


def datetime_utc_to_time(
    year,
    month = 1,
    day = 1,
    hour = 0,
    minute = 0,
    second = 0,
    microsecond = 0,
):
  """Converts a date and time in UTC to seconds since Epoch."""
  return datetime_to_time(
      UTC_TIMEZONE, year, month, day, hour, minute, second, microsecond
  )


def datetime_japan_to_time(
    year,
    month = 1,
    day = 1,
    hour = 0,
    minute = 0,
    second = 0,
    microsecond = 0,
):
  """Converts a date and time in Japan timezone to seconds since Epoch."""
  return datetime_to_time(
      JAPAN_TIMEZONE, year, month, day, hour, minute, second, microsecond
  )


def datetime_pst_to_time(
    year,
    month = 1,
    day = 1,
    hour = 0,
    minute = 0,
    second = 0,
    microsecond = 0,
):
  """Converts a date and time in PST to seconds since Epoch."""
  return datetime_to_time(
      PS_TIMEZONE, year, month, day, hour, minute, second, microsecond
  )
