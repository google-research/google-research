# Copyright 2021 The Google Research Authors.
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
: 'Run sumo experiment for city evacuation.

Example usage:
Paradise evacuation simulations.
Scenario 1. Reversed roads without traffic lights with automatic rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_0.7_test \
    --demand_file demands/demands_taz_std_0.7.rou.xml \
    --map_file map/paradise_typed_RevRd_noTFL.net.xml \
    --exit_taz map/exit_reverse.taz.add.xml \
    --detector_option paradise_reversed \
    --auto_rerouting

Scenario 2. Reversed roads with traffic lights and automatic rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_1.5_portion_0.5 \
    --demand_file demands/demands_taz_std_1.5_portion_0.5.rou.xml \
    --map_file map/paradise_typed_RevRd.net.xml \
    --exit_taz map/exit_reverse.taz.add.xml \
    --detector_option paradise_reversed \
    --auto_rerouting

Scenario 3. Baseline map with automatic rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_1.5_portion_1.5_test \
    --demand_file demands/demands_taz_std_1.5_portion_0.5.rou.xml \
    --map_file map/paradise_typed.net.xml \
    --exit_taz map/exit.taz.add.xml \
    --detector_option paradise_normal \
    --auto_rerouting

Scenario 4. Baseline map with prefixed shortest paths.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_0.11 \
    --demand_file demands/demands_shortest_path_std_0.4.rou.xml \
    --map_file map/paradise_typed.net.xml \
    --detector_option paradise_normal

Scenario 5. Add road blocker to the scenario 1, i.e. Reversed roads without
    traffic lights with automatic rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_0.7_road_block_3600 \
    --demand_file demands/demands_taz_std_0.7.rou.xml \
    --map_file map/paradise_typed_RevRd_noTFL_road_block.net.xml \
    --exit_taz map/exit_reverse.taz.add.xml \
    --road_blocker map/road_blocker_3600.rerouter.add.xml \
    --detector_option paradise_reversed \
    --auto_rerouting

Mill Valley evacuation simulations.
Scenario 1. Normal map with predetermined shortest paths.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_0.5_portion_1 \
    --demand_file demands/demands_shortest_path_std_0.5_portion_1.rou.xml \
    --map_file map/millvalley.net.xml \
    --detector_option millvalley

Scenario 2. Normal map with automaitc rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_0.5_portion_1_test \
    --demand_file demands/demands_taz_std_0.5_portion_1.rou.xml \
    --map_file map/millvalley.net.xml \
    --exit_taz map/exit.taz.add.xml \
    --detector_option millvalley \
    --auto_rerouting

Scenario 3. Reversed map no traffic lights with automaitc rerouting.
run_sumo_evacuation_sim.sh \
    --output_dir output_std_1.0_portion_0.5 \
    --demand_file demands/demands_taz_std_1.0_portion_0.5.rou.xml \
    --map_file map/millvalley_RevRd_noTFL.net.xml \
    --exit_taz map/exit.taz.add.xml \
    --detector_option millvalley \
    --auto_rerouting
'

source gbash.sh || exit 1
script_dir=""
source module "${script_dir}create_detectors.sh"


set -e

DEFINE_string --required output_dir "" \
  "Output directory. E.g. output_std_1.5_portion_0.5"
DEFINE_string --required demand_file "" \
  "Demands file. E.g. demands/demands_taz_std_1.5_portion_0.5.rou.xml"
DEFINE_string --required map_file "" \
  "Map file. E.g. map/paradise_typed_RevRd_noTFL.net.xml"
DEFINE_string exit_taz "" \
  "Evacuation exit TAZ file. E.g. map/exit.taz.add.xml"
DEFINE_string road_blocker "" \
  "Evacuation exit TAZ file. E.g. map/road_blocker.rerouter.add.xml"
DEFINE_string detector_option "" \
  "Which detector file to use."
DEFINE_bool auto_rerouting false \
  "Whether to use automatically route the cars."

gbash::init_google "$@"
set -u

# Appends a comma infront of the additional TAZ file, which is required for sumo
# with multiple arguments.
if [[ "$FLAGS_exit_taz" = ""  ]]; then
  exit_taz_file=""
else
  exit_taz_file=",${FLAGS_exit_taz}"
fi

# Appends a comma infront of the additional rerouter file, which is required for
# sumo with multiple arguments.
if [[ "$FLAGS_road_blocker" = ""  ]]; then
  road_blocker_file=""
else
  road_blocker_file=",${FLAGS_road_blocker}"
fi

date_suffix=$(date +"%Y_%m_%d_%H_%M_%S")
echo "Time prefix of this run: ${date_suffix}"

# Sets up the output folder.
if [[ -d "${FLAGS_output_dir}" ]]; then
  echo "Warning: Output folder ${FLAGS_output_dir} already exits."
  while true; do
    read -p "Do you want to overwrite the results? [Y/n]" yn
    case $yn in
      [Yy]* | '' ) echo "The new data will overwrite the results."; break;;
      [Nn]* ) echo "Quit."; exit;;
      * ) echo "Please answer yes or no. [Y/n]";;
    esac
  done
else
  mkdir "${FLAGS_output_dir}"
fi

# Sets up the output detector folder.
if [[ -d "${FLAGS_output_dir}/detector" ]]; then
  echo "Warning: Detector folder ${FLAGS_output_dir}/detector already exits."
  while true; do
    read -p "Do you want to overwrite the results? [Y/n]" yn
    case $yn in
      [Yy]* | '' ) echo "The new data will overwrite the results."; break;;
      [Nn]* ) echo "Quit."; exit;;
      * ) echo "Please answer yes or no. [Y/n]";;
    esac
  done
else
  mkdir "${FLAGS_output_dir}"/detector
fi

# Sets up the detector sumo configuration file.
create_detectors "${FLAGS_detector_option}" > \
    "${FLAGS_output_dir}/detector.add.xml"

# The probability for a vehicle to have a 'rerouting' device; default: -1
declare -a auto_rerouting_args;
if (( FLAGS_auto_rerouting )); then
  auto_rerouting_args=(
      "--device.rerouting.adaptation-steps" 5
      "--device.rerouting.with-taz"
      "--device.rerouting.adaptation-interval" 60
      "--device.rerouting.pre-period" 60)
else
  auto_rerouting_args=("--device.rerouting.probability" -1)
fi

echo ""
echo "Inputs"
echo "               Demands: ${FLAGS_demand_file}."
echo "                   Map: ${FLAGS_map_file}."
echo "              Exit TAZ: ${FLAGS_exit_taz}"
echo "          Road blocker: ${FLAGS_road_blocker}"
echo "       Detector option: ${FLAGS_detector_option}"
echo "   Automatic rerouting: ${FLAGS_auto_rerouting}"
echo ""
echo "Outputs"
echo "         Output folder: ${FLAGS_output_dir}."
echo "Detector output folder: ${FLAGS_output_dir}/detector."
echo ""

# Creates the SUMO configuration file first without running.
sumo -C run."${date_suffix}".sumocfg \
    --net-file "${FLAGS_map_file}" \
    --route-files "${FLAGS_demand_file}" \
    --additional-files "${FLAGS_output_dir}/detector.add.xml${exit_taz_file}${road_blocker_file}" \
    --fcd-output "${FLAGS_output_dir}/traffic.fcd.xml" \
    --summary-output "${FLAGS_output_dir}/summary.xml" \
    --tripinfo-output "${FLAGS_output_dir}/tripinfo.xml" \
    --vehroute-output "${FLAGS_output_dir}/vehicle_route.xml" \
    --vehroute-output.cost \
    --vehroute-output.route-length \
    --lanechange-output "${FLAGS_output_dir}/lane_change.xml"\
    --lanechange-output.started \
    --lanechange-output.ended \
    --ignore-route-errors \
    --collision.action none \
    --time-to-teleport 1200 \
    --step-length 1 \
    --end 43200 \
    --message-log "${FLAGS_output_dir}/message.log" \
    --log "${FLAGS_output_dir}/sumo.log" \
    "${auto_rerouting_args[@]}"

# Note that if you want to replicate the result, take the backup configuration
# file out of the output folder to the parent folder, then run the commend
# $ sumo -c  "run.${date_suffix}.sumocfg". It is better to rename the original
# output folder first, otherwise rerunning the experiment overwrites the
# original files in that output folder.
cputil=""

${cputil} cp "run.${date_suffix}.sumocfg" "${FLAGS_output_dir}"
echo ""
echo "The configuration file \"run.${date_suffix}.sumocfg\" has been created."
echo ""

# Run now or run later.
while true; do
  read -p "Run sumo [S], run sumo-gui [g], or later [n]?" answer
  case $answer in
    [Ss]* | '' )
        sumo -c "run.${date_suffix}.sumocfg"
        rm run."${date_suffix}".sumocfg
        break;;
    [Gg]* )
        sumo-gui -c "run.${date_suffix}.sumocfg" &
        break;;
    [Nn]* )
        echo "You can run one of the commands to start the simulation."
        echo "$ sumo -c run.${date_suffix}.sumocfg"
        echo "$ sumo-gui -c run.${date_suffix}.sumocfg"
        exit;;
    * ) echo "Please select [S/g/n].";;
  esac
done
