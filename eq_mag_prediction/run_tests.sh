# Copyright 2024 The Google Research Authors.
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

#!/bin/bash

# Function to find all test files recursively
find_test_files() {
  find "$1" -type f -name "*_test.py"
}

# Get the target directory from the first argument (optional)
target_dir="${1:-.}"

# Check if the directory exists
if [ ! -d "$target_dir" ]; then
  echo "Error: Directory '$target_dir' does not exist."
  exit 1
fi

# Find all test files
test_files=$(find_test_files "$target_dir")

# Check if any test files were found
if [ -z "$test_files" ]; then
  echo "No test files found in '$target_dir'."
else
  echo "Running tests:"
  # Generate log file name with current date
  log_file="$target_dir/tests.log"
  > "$log_file"
  # Loop through each test file and run it with absltest, redirecting output to log file
  for file in $test_files; do
    echo "TEST FILE:" $>> "$log_file"
    echo "$file" &>> "$log_file"
    python3 "$file" &>> "$log_file"
    echo "===============================================================================" &>> "$log_file"
    echo " " &>> "$log_file"
  done
fi
