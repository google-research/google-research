cmake_minimum_required(VERSION 3.10)

project(explainable_routing)

# Abseil requires C++14
set(CMAKE_CXX_STANDARD 17)

# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_library(plotly_path_string
	STATIC
		plotly_path_string.cc
		plotly_path_string.h
)
target_link_libraries(plotly_path_string
	PUBLIC
		graph
		absl::check
		absl::log
		absl::strings
)

add_library(file_util
	STATIC
		file_util.cc
		file_util.h
)
target_link_libraries(file_util
	PUBLIC
		tsv_utils
		absl::flat_hash_map
		absl::log
		absl::status
		absl::strings
)


add_library(graph
	STATIC
		graph.cc
		graph.h
)
target_link_libraries(graph
	PUBLIC
		tsv_utils
		absl::check
		absl::flat_hash_map
		absl::statusor
		absl::strings
)


add_library(neg_weight_graph_search
	STATIC
		neg_weight_graph_search.cc
		neg_weight_graph_search.h
)
target_link_libraries(neg_weight_graph_search
	PUBLIC
		graph
		link_cut_tree
		absl::flat_hash_set
		absl::log
		absl::statusor
)


add_library(path_store
	STATIC
		path_store.cc
		path_store.h
)
target_link_libraries(path_store
	PUBLIC
		tsv_utils
		formulation
		graph
		neg_weight_graph_search
		absl::check
		absl::log
		absl::statusor
)


add_library(link_cut_tree
	STATIC
		link_cut_tree.h
		link_cut_tree.cc
)
target_link_libraries(link_cut_tree
	PUBLIC
)


add_library(formulation
	INTERFACE
		formulation.h
)
target_link_libraries(formulation
	INTERFACE
		graph
		absl::flat_hash_map
		absl::flat_hash_set
)

add_library(penalty_runner
	STATIC
		penalty_runner.cc
		penalty_runner.h
)
target_link_libraries(penalty_runner
	PUBLIC
		plotly_path_string
		graph
		neg_weight_graph_search
		path_store
		absl::check
		absl::flat_hash_map
		absl::flat_hash_set
		absl::log
		absl::statusor
		absl::strings
		absl::time
)

add_library(residual_problem
	STATIC
		residual_problem.cc
		residual_problem.h
)
target_link_libraries(residual_problem
	PUBLIC
		formulation
		graph
		neg_weight_graph_search
		path_store
		plotly_path_string
		absl::check
		absl::flat_hash_map
		absl::flat_hash_set
		absl::log
		absl::statusor
)

add_executable(explainable_routing_scenarios_main explainable_routing_scenarios_main.cc)
target_link_libraries(explainable_routing_scenarios_main
	PUBLIC
		plotly_path_string
		file_util
		formulation
		graph
		path_store
		penalty_runner
		residual_problem
		absl::check
		absl::flags
		absl::flags_parse
		absl::flat_hash_map
		absl::flat_hash_set
		absl::log
		absl::statusor
		absl::strings
		absl::time
)

add_library(tsv_utils
	STATIC
		tsv_utils.cc
		tsv_utils.h
)
target_link_libraries(tsv_utils
	PUBLIC
		absl::flat_hash_map
		absl::check
		absl::strings
)
