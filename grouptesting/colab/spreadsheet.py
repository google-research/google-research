# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Connect to and interact with google spreadsheet."""

import datetime
from typing import Optional

import numpy as np


def index_to_column(idx):
  """Turns an index into a letter representing a spreadsheet column."""
  letter = chr(65 + idx % 26)
  q = idx // 26
  if q == 0:
    return letter
  else:
    return index_to_column(q-1) + letter


def create_template_sheet(gc, name, params, infection_rate = 0.05):
  """Creates an empty sheet made of 3 worksheet: params, prior and groups."""
  sheet = gc.create(name)

  # Params
  worksheet = sheet.sheet1
  worksheet.resize(6, 2)
  num_params = len(params)
  cells = worksheet.range(f'A1:B{num_params}')
  list_params = list(params.items())
  for cell in cells:
    cell.value = list_params[cell.row - 1][cell.col - 1]
  worksheet.update_cells(cells)

  # Priors
  num_patients = params['patients']
  sheet.add_worksheet('priors', num_patients + 1, 2)
  worksheet = sheet.get_worksheet(1)
  cells = worksheet.range(f'A1:B{num_patients+1}')
  for cell in cells:
    if cell.row == 1:
      cell.value = '' if cell.col == 1 else 'Prior'
    else:
      default_value = infection_rate
      cell.value = f'Patient {cell.row}' if (cell.col == 1) else default_value
  worksheet.update_cells(cells)

  offset = 2
  sheet.add_worksheet('groups', num_patients + offset, 1)
  worksheet = sheet.get_worksheet(2)
  cells = worksheet.range(f'A2:A{num_patients + offset}')
  for cell in cells:
    if cell.row > offset:
      cell.value = f'Patient {cell.row - offset}'
    else:
      cell.value = 'Result'
  worksheet.update_cells(cells)
  worksheet.freeze(rows=offset, cols=0)


def extract(worksheet,
            start_row,
            start_col,
            end_row,
            end_col):
  """Extracts a range given as zero-based indices from a spreadsheet."""
  start_row = start_row + 1  # zero-based vs. one-based indexing
  end_row = start_row if end_row is None else end_row + 1
  start_char = index_to_column(start_col)
  end_char = index_to_column(end_col)
  cell_range = f'{start_char}{start_row}:{end_char}{end_row}'

  result = np.zeros((end_row - start_row + 1, end_col - start_col + 1))
  cells = worksheet.range(cell_range)
  for cell in cells:
    col = cell.col - 1 - start_col
    row = cell.row - 1 - start_row
    result[row, col] = cell.value
  return np.array(result)


class GroupTestingSheet:
  """Create, read and write into a spreadsheet."""

  def __init__(self,
               google_credentials,
               sheet_id = None,
               copy_from_id = None):
    """Initialization.

    Args:
      google_credentials: a google credentials object returned by
        gspread.authorize to manipulate spreadsheets.
      sheet_id: an optional str representing an existing spreadsheet to be used.
      copy_from_id: an optional str representing a template spreadsheet to be
        copied.
    """
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M')
    name = f'Group Testing - {now}'
    # Initialize with some default values.
    self.params = dict(
        patients=10, sensitivity=0.97, specificity=0.95,
        tests_per_cycle=8, cycles=10, max_group_size=6)
    self._gc = google_credentials
    if sheet_id is not None:
      self.sheet = self._gc.open_by_key(sheet_id)
    elif copy_from_id is not None:
      self.sheet = self._gc.copy(file_id=copy_from_id, title=name)
    else:
      raise ValueError('One of `sheet_id` or `copy_from_id` should be not None')
    self.load()

  @property
  def num_patients(self):
    return self.params['patients']

  def load(self):
    """Reads the spreadsheet, loads and adapt the data."""
    worksheet = self.sheet.sheet1
    num_params = len(self.params)
    cells = worksheet.range(f'A1:B{num_params}')
    params = [[None, None] for _ in range(num_params)]
    for cell in cells:
      if cell.col == 1:
        value = cell.value
      else:
        cast = float if '.' in cell.value else int
        value = cast(cell.value)
      params[cell.row - 1][cell.col - 1] = value
    self.params = dict(params)

    # Adapts the other two worksheets
    offsets = [1, 2]
    num_patients = self.num_patients
    for wid, offset in zip([1, 2], offsets):
      worksheet = self.sheet.get_worksheet(wid)
      num_cols = max(2, worksheet.col_count)
      worksheet.resize(num_patients + offset, num_cols)
    worksheet = self.sheet.get_worksheet(0)
    worksheet.update_acell('B1', num_patients)

  @property
  def priors(self):
    """Reads the array of prior per patient from the spreadsheet."""
    def parse(value):
      return float(value.rstrip('%')) / 100.0

    wks = self.sheet.get_worksheet(1)
    max_row = self.num_patients + 1
    return np.array([parse(cell.value) for cell in wks.range(f'B2:B{max_row}')])

  def write_groups(self, groups):
    """Adds the groups boolean array to the proper worksheet."""
    worksheet = self.sheet.get_worksheet(2)
    initial_num_groups = worksheet.col_count - 1
    first_time = initial_num_groups == 1

    num_groups, num_patients = groups.shape
    start = worksheet.col_count
    end = start + num_groups
    worksheet.add_cols(num_groups)
    groups = groups.astype(np.int32).tolist()
    start_char, end_char = index_to_column(start), index_to_column(end - 1)

    # Add header
    cells = worksheet.range(f'{start_char}1:{end_char}1')
    for cell in cells:
      cell.value = f'g{cell.col - 1 - int(first_time)}'
    worksheet.update_cells(cells)

    # Add groups
    offset = 2
    cell_range = f'{start_char}{1 + offset}:{end_char}{num_patients + offset}'
    cells = worksheet.range(cell_range)
    for cell in cells:
      cell.value = groups[cell.col - start - 1][cell.row - offset - 1]
    worksheet.update_cells(cells)

    if first_time == 1:
      worksheet.delete_columns(2)

  def write_marginals(self, marginals):
    """Writes the marginal in the spreadsheet, next to the priors."""
    worksheet = self.sheet.get_worksheet(1)
    if worksheet.col_count < 3:
      worksheet.add_cols(1)

    num_patients = self.params['patients']
    cell_range = f'C1:C{num_patients + 1}'
    cells = worksheet.range(cell_range)
    marginals = marginals.tolist()
    for cell in cells:
      cell.value = marginals[cell.row - 2] if cell.row > 1 else 'Marginal'
    worksheet.update_cells(cells)

  def _read_bools(self,
                  start_row,
                  end_row = None,
                  last_cols = None):
    """Returns a np.ndarray of boolean extracted from the groups spreadsheet.

    Args:
      start_row: the starting row of the boolean matrix.
      end_row: the last row of the boolean matrix. If not set, we only read the
        start_row.
      last_cols: if not None the number of columns to read, else we read all of
        them.

    Returns:
      A np.ndarray<bool>[end_row - start_row + 1, last_cols]
    """
    worksheet = self.sheet.get_worksheet(2)
    end_row = start_row if end_row is None else end_row
    num_cols = worksheet.col_count - 1 if last_cols is None else last_cols
    start_col = worksheet.col_count - num_cols
    end_col = worksheet.col_count - 1
    result = extract(worksheet, start_row, start_col, end_row, end_col)
    return result.astype(np.bool)

  def read_groups(self, num_groups = None):
    """Reads the last num_groups from the spreadsheet.

    Args:
      num_groups: the number of groups for which we want to get the results. If
        not set, we read all of them.
    Returns:
      A np.ndarray<bool>[num_groups, num_patients].
    """
    worksheet = self.sheet.get_worksheet(2)
    num_groups = worksheet.col_count - 1
    if num_groups < 2:
      return np.array([])

    offset = 2
    result = self._read_bools(
        offset, offset + self.num_patients - 1, num_groups)
    return result.T

  def read_results(self, num_groups = None):
    """Reads the last num_groups results from the spreasheet.

    Args:
      num_groups: the number of groups for which we want to get the results. If
        not set, we read all of them.
    Returns:
      A np.ndarray<bool>[num_groups].
    """
    return np.squeeze(self._read_bools(1, 1, num_groups))

  @property
  def groups_url(self):
    gid = self.sheet.get_worksheet(2).id
    return f'{self.sheet.url}/edit#gid={gid}'
