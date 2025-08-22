Introduction


This is the code for the experiments in "First Passage Percolation with
Queried Hints." The experiment takes a snapshot of traffic and approximates
distances between pairs of points within a given query set by using an independently-sampled (resampled) snapshot of traffic, probing a
carefully-chosen set of edges from the original snapshot, and running Dijkstra.


Reproducing the points in the plots


1. Install Abseil with CMake: https://abseil.io/docs/cpp/quickstart-cmake
2. Download the directory containing this README file to "~/Code"
(or a directory of your choice)
3. Run "cd ~/Code", followed by
"git clone https://github.com/abseil/abseil-cpp.git"
4. In the ~/Code directory, run "mkdir build" and "cd build"
5. Unzip https://storage.googleapis.com/gresearch/probe-routing/osm_graphs.zip
to make ~/osm_graphs.
6. In random_weights_main.cc, change the tsv_directory flag value to the
location of the baden-processed or washington-processed folder, depending on
which region you want to run.
7. Now, we are in ~/Code/build. Run "cmake -DCMAKE_BUILD_TYPE=Release .." to
prepare to build the code. The Release build type is chosen to make sure that
compiler optimizations are applied, which makes file loading much faster.
8. Run "cmake --build . --target random_weights_main" to build the code.
9. Run "./random_weights_main". This should take at most 20 minutes.

This produces all of the data required. To aggregate the data into a plot, do
the following:

10. Copy all output after the word "DONE" to a file. See washington_results.txt
for an example.
11. Enter a python interpreter by typing "python3."
12. Type "import extract_traffic_results as etr" and "etr.make_max_rcf_plot('washington_results.txt')" to produce a plot.


Rerunning the entire experiment

The steps outlined before rely on the following precomputed data, all in the osm_graphs folder:

a. A traffic snapshot (highway_traffic_2.0)
b. An independently-sampled snapshot (resampled_traffic_2.0)
c. A set of queries (medium_queries_random_100.tsv)
d. Cached shortest paths computed via Dijkstra in various graphs
(path_cache__...)

a,b, and c are random and are generated using graphs_with_highway_traffic.py
(for a and b) and generate_random_queries.py (for c). Given a,b, and c, d is deterministic and is only provided to speed up the code. One can delete the path_cache__ files and the code will automatically regenerate them,
albeit slowly.