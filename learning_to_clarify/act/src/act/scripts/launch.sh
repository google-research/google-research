#!/bin/bash
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



UNKNOWN_USAGE="Unknown usage: $0 $@"

USAGE="Usage: $0 <command> <argument1> <argument2> ...\n
\n
Supported commands:\n
 - run-act\n
 - run-sft\n
 - generate-preference\n
 - evaluate\n
\n
Example: $0 run-act --config=gs://learning-to-clarify-staging/demo_config.json"

case $1 in
  run-act)
    shift
    COMMAND="python -m accelerate.commands.launch --config_file ./deepspeed_zero3.yaml -m act.scripts.run_act $@"
    ;;
  run-sft)
    shift
    COMMAND="python -m accelerate.commands.launch --config_file ./deepspeed_zero3.yaml -m act.scripts.run_sft $@"
    ;;
  generate-preference)
    shift
    COMMAND="python -m act.scripts.generate_preference $@"
    ;;
  evaluate)
    shift
    COMMAND="python -m act.scripts.evaluate $@"
    ;;
  *)
    echo -e $UNKNOWN_USAGE
    echo -e $USAGE
    exit 1
    ;;
esac

exec $COMMAND
