# Evacuation Traffic Simulation Tool Kit

This tool kit works with SUMO.

The process to run an evacuation simulation.

1.  Get the map. The maps are located in Paradise_template/map or
    MillValley_template/map. They are first downloaded using OSM, then converted
    to the sumo net using NETCONVERT. The maps are recommended to put under the
    scenario folder.

2.  Explore the map data. Command example. example output:

    <!--
    Total number of all edges: 4957
    Residential road stats
    Sum of lengths:  321047.69
    Max length:  1832.7
    Min length:  0.1
    # edges < 1 / 0.0415:  390
    Sum of residential roads:  2736

    Services road stats
    Sum of lengths:  34401.42
    Max length:  309.37
    Min length:  0.1
    # edges < 1 / 0.0415:  530
    Sum of residential roads:  1057
    -->

3.  Get the demands. The demands are created using the model
    //research/simulation/traffic:random_traffic_generator. The demands are
    recommended to put under the scenario folder.

4.  Run the simulation. Use the script
    //research/simulation/traffic/run_sumo_evacuation_sim.sh. The docstring in
    the script provides the commands to run the simulation.

5.  Get the speedmap. 4.a. Parse the FCD file from the output folder, and cut
    the data into different time ranges. 4.b. Generate the pdf output figures
    for each time range. 4.c. Compile the jpg files into the gif file using the
    script //research/simulation/traffic/pdf_to_images.sh

6.  Get the demands-evacuation curves.

7.  The required data files are available here:  https://storage.cloud.google.com/traffic-sim/
