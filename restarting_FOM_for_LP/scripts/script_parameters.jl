# This file is not meant to be run. It just contains parameters for the scripts.
ALL_PROBLEM_NAMES = ["qap10", "qap15", "nug08-3rd", "nug20"]

PLOTS_KKT_TOLERANCE = 1e-9
TABLE_OF_RESULTS_TARGET_TOLERANCE = 1e-6

ITERATION_LIMIT_FOR_FINDING_BEST_PRIMAL_WEIGHT = 5000

# restart lengths to try for each problem
RESTART_LENGTHS_DICT = Dict(
  "qap10" => 4 .^ collect(1:9),
  "qap15" => 4 .^ collect(1:9),
  "nug08-3rd" => 4 .^ collect(1:9),
  "nug20" => 4 .^ collect(1:9),
)

# iteration limits
ITERATION_LIMIT_DICT = Dict(
  "qap10" => Dict(PDHG => 500000, EXTRAGRADIENT => 500000, ADMM => 60000),
  "qap15" => Dict(PDHG => 500000, EXTRAGRADIENT => 500000, ADMM => 60000),
  "nug08-3rd" => Dict(PDHG => 10000, EXTRAGRADIENT => 10000, ADMM => 10000),
  "nug20" => Dict(
    PDHG => 500000,
    EXTRAGRADIENT => 500000,
    ADMM => 150000,
  ),
)
