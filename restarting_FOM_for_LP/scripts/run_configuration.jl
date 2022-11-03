import FirstOrderLp
import CSV

include("../src/PrimalDualMethods.jl")
include("script_parameters.jl")
include("../src/choose_primal_weight.jl")

# Parse command line arguements
@assert length(ARGS) == 9

test_problem_folder = ARGS[1]

results_csv_file = ARGS[2]
@assert length(results_csv_file) > 3
@assert results_csv_file[end-3:end] == ".csv"

method = RecoverPrimalDualMethodFromString(ARGS[3])

problem_name = ARGS[4]


restart_scheme_name_arg = ARGS[5]

restart_length = parse(Int64, ARGS[6])

if ARGS[7] == "yes"
  always_reset_to_average = true
elseif ARGS[7] == "no"
  always_reset_to_average = false
else
  error("unknown value for reset to average")
end

if restart_scheme_name_arg == "no_restarts"
  restart_scheme = NoRestarts()
elseif restart_scheme_name_arg == "fixed_frequency"
  restart_scheme =
    FixedFrequencyRestarts(restart_length, always_reset_to_average)
elseif restart_scheme_name_arg == "adaptive"
  restart_scheme = AdaptiveRestarts(exp(-1), always_reset_to_average)
else
  error("unknown restart scheme")
end

iteration_limit = parse(Int64, ARGS[8])

kkt_tolerance = parse(Float64, ARGS[9])


println("OPTIONS SELECTED:")
@show test_problem_folder
@show results_csv_file
@show method
@show problem_name
@show restart_scheme
@show iteration_limit
@show kkt_tolerance

params = PrimalDualOptimizerParameters(
  method,
  nothing, # step_size (forces the solver to use a provably correct step size)
  1.0, # primal_weight
  30, # record every
  100, # print every
  true, # verbose
  ITERATION_LIMIT_FOR_FINDING_BEST_PRIMAL_WEIGHT, # iteration limit
  NoRestarts(), # restart scheme
  nothing, # don't record information on subspace
  kkt_tolerance,
)

instance_path = joinpath(test_problem_folder, "$(problem_name).mps.gz")
lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
lp.variable_upper_bound .= Inf

if method == ADMM
  convert_problem_to_standard_form!(lp)
end

println("FIND BEST PRIMAL WEIGHT")
best_primal_weight =
  find_best_primal_weight(params, lp, 4.0 .^ collect(-5:5), 1.0, 1.0)
println("best_primal_weight = $best_primal_weight")

println("RUN ALGORITHM")
params.restart_scheme = restart_scheme
params.iteration_limit = iteration_limit
params.primal_weight = best_primal_weight
solver_output = optimize(params, lp)
CSV.write(results_csv_file, solver_output.iteration_stats)
