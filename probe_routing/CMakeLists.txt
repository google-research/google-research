cmake_minimum_required(VERSION 3.10)

project(aistats_traffic)

# Abseil requires C++14
set(CMAKE_CXX_STANDARD 17)

# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_library(random_weights_utils
	STATIC
		random_weights_utils.cc
		random_weights_utils.h
)
target_link_libraries(random_weights_utils
	PUBLIC
		graph
	    graph_search
	    graph_utils
	    tsv_utils
	    absl::flat_hash_map
	    absl::flags
	    absl::log
	    absl::check
	    absl::status
	    absl::statusor
	    absl::strings
)


add_executable(random_weights_main random_weights_main.cc)
target_link_libraries(random_weights_main
	PUBLIC
		file_util
        graph
        graph_search
        graph_utils
        random_weights_utils
        tsv_utils
        absl::log_severity
        absl::flat_hash_map
        absl::flat_hash_set
        absl::flags
        absl::log
        absl::check
        absl::random_random
        absl::status
        absl::statusor
        absl::strings
)


add_library(file_util
	STATIC
		file_util.cc
		file_util.h
)
target_link_libraries(file_util
	PUBLIC
		graph
		tsv_utils
)


add_library(graph
	STATIC
		graph.cc
		graph.h
)
target_link_libraries(graph
	PUBLIC
		tsv_utils
		absl::flat_hash_map
		absl::strings
)


add_library(graph_search
	STATIC
		graph_search.cc
		graph_search.h
)
target_link_libraries(graph_search
	PUBLIC
		graph
		graph_utils
		tsv_utils
		absl::flat_hash_set
		absl::log
		absl::statusor
)


add_library(graph_utils
	STATIC
		graph_utils.cc
		graph_utils.h
)
target_link_libraries(graph_utils
	PUBLIC
		graph
		absl::check
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
