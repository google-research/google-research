CXXFLAGS=-std=c++20 -O3 -Wall -Wextra

all: exposure_design_main

test: exposure_design_test weighted_distribution_test
	./exposure_design_test
	./weighted_distribution_test

.PHONY: all test

exposure_design_main: exposure_design_main.cc exposure_design.cc exposure_design.h exposure_design_internal.h weighted_distribution.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< exposure_design.cc

exposure_design_test: exposure_design_test.cc exposure_design.cc exposure_design.h exposure_design_internal.h weighted_distribution.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< exposure_design.cc

weighted_distribution_test: weighted_distribution_test.cc weighted_distribution.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $<
