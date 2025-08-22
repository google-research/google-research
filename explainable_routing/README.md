This is the code for the experiments in "Why is My Route Different Today?
An Algorithm for Explaining Route Selection." The experiment generates
explanations for routes in a collection of synthetically-generated
traffic scenarios.

1. Install Abseil with CMake: https://abseil.io/docs/cpp/quickstart-cmake
2. Download the directory containing this README file to "~/Code"
(or a directory of your choice)
3. Run "cd ~/Code", followed by
"git clone https://github.com/abseil/abseil-cpp.git"
4. In the ~/Code directory, run "mkdir build" and "cd build"
5. Extract the osm_graphs directory from
http://storage.googleapis.com/gresearch/explainable-routing/osm_graphs.zip
to ~/osm_graphs.
6. In explainable_routing_scenarios_main.cc, change the tsv_directory flag
value to the location of the baden-processed or washington-processed folder,
depending on which region you want to run.
7. Now, we are in ~/Code/build. Run "cmake -DCMAKE_BUILD_TYPE=Release .." to
prepare to build the code. The Release build type is chosen to make sure that
compiler optimizations are applied, which makes file loading much faster.
8. Run "cmake --build . --target explainable_routing_scenarios_main" to
build the code.
9. Run "./explainable_routing_scenarios_main". This can take a while but should
be very fast if subgraphs have been cached already in the osm_graphs directory.

This produces all of the data required. To view the output:

10. Go to osm_graphs/washington-processed/explainable_output
11. Go to any subdirectory
12. Pick any plotly_*.txt file and run
"python3 draw_as_plotly.py plotly_BLAH.txt" to generate a maps image.
13. Get a birds-eye view of which query pairs exist by running
"python3 draw_as_plotly.py
summary_plotly__seattle_medium_queries_random_300.txt".
14. Each subdirectory of the explainable_output folder contains all outputs from
a given parameter range.
summary_stats__seattle_medium_queries_random_300.txt has aggregate information
about how small explanations are.

The steps outlined before rely on the following precomputed data, all in the osm_graphs folder:

a. The graph (nodes.tsv and arcs.tsv)
b. A set of queries (medium_queries_random_100.tsv)

Cached data speeds up the experiment. path_cache_* are the most effective subdirectories for speeding things up.
Cycle and cut caches are not as necessary.
Query sets can be generated using generate_random_queries.py.
The tables in the paper can be generated using get_exp_subgraph_fracs.py.
To run get_exp_subgraph_fracs.py, you'll either need to run
explainable_routing_scenarios_main on a variety of parameters, or you'll
need to download the cut caches (present in
http://storage.googleapis.com/gresearch/explainable-routing/big_osm_graphs.zip)
and run get_exp_subgraphs_fracs.py on that osm_graphs directory.

If you want to time explanation generation, make sure to delete cached data
for the appropriate parameter set before running. Otherwise, running the
code will time cache reloading time only.