# Copyright 2022 The Google Research Authors.
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
: 'Generates the detector configuraiton file.

The detectors can be extended for more cases as long as users know where to
place them, how frequently to dump reports, etc. Details about the SUMO
detectors can be found at:
https://sumo.dlr.de/docs/Simulation/Output/Induction_Loops_Detectors_(E1).html
Note that the detector data output director is relative the location of the
detector configuration file, so the output data files in the following settings
`file="detector/e1Detector..."` treat the location of the detector file as the
root directory.
'

create_detectors()
{

# Create detectors for Paradise evacuation simulation. The detectors are placed
# not only at the normal exits, but also on the revesed roads extis.
if [[ "$1" = "paradise_reversed" ]]; then

echo """<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<additional>

    <!-- Clark Rd -->
    <e1Detector id=\"e1Detector_-184839999#0_0_3\" lane=\"-184839999#0_0\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_0_3.xml\"/>
    <e1Detector id=\"e1Detector_-184839999#0_1_4\" lane=\"-184839999#0_1\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_1_4.xml\"/>
    <!-- Clark Rd reversed -->
    <e1Detector id=\"e1Detector_-184839999#0_2_8\" lane=\"-184839999#0_2\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_2_8.xml\"/>
    <e1Detector id=\"e1Detector_-184839999#0_3_9\" lane=\"-184839999#0_3\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_3_9.xml\"/>

    <!-- Pentz Rd -->
    <e1Detector id=\"e1Detector_-538864403#0_0_5\" lane=\"-538864403#0_0\" pos=\"150.00\" freq=\"60.00\" file=\"detector/e1Detector_-538864403#0_0_5.xml\"/>
    <!-- Pentz Rd reversed -->
    <e1Detector id=\"e1Detector_-538864403#0_1_10\" lane=\"-538864403#0_1\" pos=\"150.00\" freq=\"60.00\" file=\"detector/e1Detector_-538864403#0_1_10.xml\"/>

    <!-- Neal Rd -->
    <e1Detector id=\"e1Detector_10293408#4_0_2\" lane=\"10293408#4_0\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_10293408#4_0_2.xml\"/>

    <!-- Skyway -->
    <e1Detector id=\"e1Detector_27323694_0_0\" lane=\"27323694_0\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_27323694_0_0.xml\"/>
    <e1Detector id=\"e1Detector_27323694_1_1\" lane=\"27323694_1\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_27323694_1_1.xml\"/>
    <!-- Skyway reversed -->
    <e1Detector id=\"e1Detector_37625137#1_0_6\" lane=\"37625137#1_0\" pos=\"120.00\" freq=\"60.00\" file=\"detector/e1Detector_37625137#1_0_6.xml\"/>
    <e1Detector id=\"e1Detector_37625137#1_1_7\" lane=\"37625137#1_1\" pos=\"120.00\" freq=\"60.00\" file=\"detector/e1Detector_37625137#1_1_7.xml\"/>

</additional>"""

# Create detectors for Paradise evacuation simulation.
elif [[ "$1" = "paradise_normal" ]]; then
echo """<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<additional>

    <!-- Clark Rd -->
    <e1Detector id=\"e1Detector_-184839999#0_0_3\" lane=\"-184839999#0_0\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_0_3.xml\"/>
    <e1Detector id=\"e1Detector_-184839999#0_1_4\" lane=\"-184839999#0_1\" pos=\"90.00\" freq=\"60.00\" file=\"detector/e1Detector_-184839999#0_1_4.xml\"/>

    <!-- Pentz Rd -->
    <e1Detector id=\"e1Detector_-538864403#0_0_5\" lane=\"-538864403#0_0\" pos=\"150.00\" freq=\"60.00\" file=\"detector/e1Detector_-538864403#0_0_5.xml\"/>

    <!-- Neal Rd -->
    <e1Detector id=\"e1Detector_10293408#4_0_2\" lane=\"10293408#4_0\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_10293408#4_0_2.xml\"/>

    <!-- Skyway -->
    <e1Detector id=\"e1Detector_27323694_0_0\" lane=\"27323694_0\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_27323694_0_0.xml\"/>
    <e1Detector id=\"e1Detector_27323694_1_1\" lane=\"27323694_1\" pos=\"30.00\" freq=\"60.00\" file=\"detector/e1Detector_27323694_1_1.xml\"/>

</additional>"""

# Create detectors for Mill Valley evacuation simulation.
elif [[ "$1" = "millvalley" ]]; then
echo """<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<additional>

    <!-- US101 South: 35869652 -->
    <e1Detector id=\"e1Detector_35869652_0_0\" lane=\"35869652_0\" pos=\"450\" freq=\"60.00\" file=\"detector/e1Detector_35869652_0_0.xml\"/>
    <e1Detector id=\"e1Detector_35869652_1_1\" lane=\"35869652_1\" pos=\"450\" freq=\"60.00\" file=\"detector/e1Detector_35869652_1_1.xml\"/>
    <e1Detector id=\"e1Detector_35869652_2_2\" lane=\"35869652_2\" pos=\"450\" freq=\"60.00\" file=\"detector/e1Detector_35869652_2_2.xml\"/>
    <e1Detector id=\"e1Detector_35869652_3_3\" lane=\"35869652_3\" pos=\"450\" freq=\"60.00\" file=\"detector/e1Detector_35869652_3_3.xml\"/>
    <e1Detector id=\"e1Detector_35869652_4_4\" lane=\"35869652_4\" pos=\"450\" freq=\"60.00\" file=\"detector/e1Detector_35869652_4_4.xml\"/>

    <!-- US101 South: 394150403 -->
    <e1Detector id=\"e1Detector_394150403_0_0\" lane=\"394150403_0\" pos=\"150\" freq=\"60.00\" file=\"detector/e1Detector_394150403_0_0.xml\"/>
    <e1Detector id=\"e1Detector_394150403_1_1\" lane=\"394150403_1\" pos=\"150\" freq=\"60.00\" file=\"detector/e1Detector_394150403_1_1.xml\"/>
    <e1Detector id=\"e1Detector_394150403_2_2\" lane=\"394150403_2\" pos=\"150\" freq=\"60.00\" file=\"detector/e1Detector_394150403_2_2.xml\"/>
    <e1Detector id=\"e1Detector_394150403_3_3\" lane=\"394150403_3\" pos=\"150\" freq=\"60.00\" file=\"detector/e1Detector_394150403_3_3.xml\"/>
    <e1Detector id=\"e1Detector_394150403_4_4\" lane=\"394150403_4\" pos=\"150\" freq=\"60.00\" file=\"detector/e1Detector_394150403_4_4.xml\"/>

    <!-- US101 North: 30682440 -->
    <e1Detector id=\"e1Detector_30682440_0_8\" lane=\"30682440_0\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_30682440_0_8.xml\"/>
    <e1Detector id=\"e1Detector_30682440_1_7\" lane=\"30682440_1\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_30682440_1_7.xml\"/>
    <e1Detector id=\"e1Detector_30682440_2_6\" lane=\"30682440_2\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_30682440_2_6.xml\"/>
    <e1Detector id=\"e1Detector_30682440_3_5\" lane=\"30682440_3\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_30682440_3_5.xml\"/>
    <e1Detector id=\"e1Detector_30682440_4_4\" lane=\"30682440_4\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_30682440_4_4.xml\"/>

    <!-- US101 North: 23797526 -->
    <e1Detector id=\"e1Detector_23797526_0_8\" lane=\"23797526_0\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_23797526_0_8.xml\"/>
    <e1Detector id=\"e1Detector_23797526_1_7\" lane=\"23797526_1\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_23797526_1_7.xml\"/>
    <e1Detector id=\"e1Detector_23797526_2_6\" lane=\"23797526_2\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_23797526_2_6.xml\"/>
    <e1Detector id=\"e1Detector_23797526_3_5\" lane=\"23797526_3\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_23797526_3_5.xml\"/>
    <e1Detector id=\"e1Detector_23797526_4_4\" lane=\"23797526_4\" pos=\"275\" freq=\"60.00\" file=\"detector/e1Detector_23797526_4_4.xml\"/>

</additional>"""
else

echo "Wrong detector option." >&2
return 1

fi
}
