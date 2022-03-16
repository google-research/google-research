###################################
# variables shared across scripts #
###################################
ALL_PROBLEM_NAMES = ["qap10", "qap15", "nug08-3rd", "nug20"]

# restart lengths to try for eahc problem
RESTART_LENGTHS_DICT = Dict(
  "qap10" => 4 .^ collect(1:9),
  "qap15" => 4 .^ collect(1:9),
  "nug08-3rd" => 4 .^ collect(1:9),
  "nug20" => 4 .^ collect(1:9),
)

# other parameter values
ITERATION_LIMIT = 500000
