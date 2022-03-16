import FirstOrderLp
import CSV

include("../src/PrimalDualMethods.jl")
include("../src/shared_variables.jl")

function find_best_primal_weight(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  primal_weight_trial_list::Vector{Float64},
  primal_weight_for_evaluation::Float64,
  step_size_for_evaluation::Float64,
)
  best_kkt_error = Inf
  best_primal_weight = NaN
  for primal_weight in primal_weight_trial_list
    println("run for primal weight = $primal_weight")
    params.primal_weight = primal_weight
    solver_output = optimize(params, problem)

    kkt_error = min(
      solver_output.iteration_stats.kkt_error_average_iterate[end],
      solver_output.iteration_stats.kkt_error_current_iterate[end],
    )

    println("kkt_error = $kkt_error")
    if kkt_error < best_kkt_error
      best_kkt_error = kkt_error
      best_primal_weight = primal_weight
    end
  end
  return best_primal_weight
end

mutable struct ResultsCsvPaths
  no_restarts::String
  adaptive_restarts::String
  dynamic_adaptive_restarts::String
  restart_lengths::Dict{Int64,String}
end

function get_csv_paths(
  results_directory::String,
  restart_length_list::Vector{Int64},
)
  restart_length_csv_paths = Dict{Int64,String}()
  for restart_length in restart_length_list
    restart_length_csv_paths[restart_length] = "$results_directory/restart_length$(restart_length).csv"
  end
  return ResultsCsvPaths(
    "$results_directory/no_restarts.csv",
    "$results_directory/adaptive_restarts.csv",
    "$results_directory/dynamic_adaptive_restarts.csv",
    restart_length_csv_paths,
  )
end

function solve_problem_with_multiple_methods(;
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  restart_length_list::Vector{Int64},
  results_directory::String,
  iteration_limit_for_finding_best_primal_weight::Int64,
  iteration_limit::Int64,
  kkt_tolerance::Float64,
  method::PrimalDualMethod,
)
  println("solving problem with: ")
  print("rows = ", size(problem.constraint_matrix, 1), ", ")
  print("cols = ", size(problem.constraint_matrix, 2), ", ")
  println("nnz = ", length(problem.constraint_matrix.nzval), ".")

  saved_stdout = stdout
  log_file = open("$results_directory/best_primal_weight.txt", "w")
  redirect_stdout(log_file)
  params = PrimalDualOptimizerParameters(
    method,
    nothing, # step_size (forces the solver to use a provably correct step size)
    1.0, # primal_weight
    30, # record every
    100, # print every
    true, # verbose
    iteration_limit_for_finding_best_primal_weight, # iteration limit
    NoRestarts(), # restart scheme
    nothing, # don't record information on subspace
    kkt_tolerance,
  )

  println("FIND BEST PRIMAL WEIGHT")
  best_primal_weight =
    find_best_primal_weight(params, problem, 4.0 .^ collect(-5:5), 1.0, 1.0)
  println("best_primal_weight = $best_primal_weight")
  close(log_file)

  csv_paths = get_csv_paths(results_directory, restart_length_list)

  log_file = open("$results_directory/no_restarts.txt", "w")
  redirect_stdout(log_file)
  params.iteration_limit = iteration_limit
  params.primal_weight = best_primal_weight
  println("Run no restarts")
  solver_output = optimize(params, problem)
  close(log_file)
  CSV.write(csv_paths.no_restarts, solver_output.iteration_stats)

  log_file = open("$results_directory/adaptive_restarts.txt", "w")
  redirect_stdout(log_file)
  println("Run adaptive restarts")
  params.restart_scheme = AdaptiveRestarts(exp(-1), true)
  solver_output = optimize(params, problem)
  close(log_file)
  CSV.write(csv_paths.adaptive_restarts, solver_output.iteration_stats)


  log_file = open("$results_directory/dynamic_adaptive_restarts.txt", "w")
  redirect_stdout(log_file)
  println("Run extra adaptive restarts")
  params.restart_scheme = AdaptiveRestarts(exp(-1), false)
  solver_output = optimize(params, problem)
  close(log_file)
  CSV.write(csv_paths.dynamic_adaptive_restarts, solver_output.iteration_stats)

  for restart_length in restart_length_list
    log_file =
      open("$results_directory/restart_length_$(restart_length).txt", "w")
    redirect_stdout(log_file)
    println("restart_length = $restart_length")
    params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
    solver_output = optimize(params, problem)
    close(log_file)
    CSV.write(
      csv_paths.restart_lengths[restart_length],
      solver_output.iteration_stats,
    )
  end
  redirect_stdout(saved_stdout)
end

@assert length(ARGS) == 4
test_problem_folder = ARGS[1]
results_directory = ARGS[2]
method = RecoverPrimalDualMethodFromString(ARGS[3])
problem_name_arg = ARGS[4]

function main()
  ################
  # run problems #
  ################
  iteration_limit_for_finding_best_primal_weight = 5000
  kkt_tolerance = 1e-9

  if problem_name_arg == "all"
    problem_name_list = ALL_PROBLEM_NAMES
  elseif problem_name_arg in ALL_PROBLEM_NAMES
    problem_name_list = [problem_name_arg]
  else
    error("unknown problem name arguement supplied")
  end

  for problem_name in problem_name_list
    instance_path = joinpath(test_problem_folder, "$(problem_name).mps.gz")
    lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
    lp.variable_upper_bound .= Inf

    results_subdirectory = joinpath(results_directory, problem_name)
    mkpath(results_subdirectory)

    solve_problem_with_multiple_methods(;
      problem = lp,
      restart_length_list = RESTART_LENGTHS_DICT[problem_name],
      results_directory = results_subdirectory,
      iteration_limit_for_finding_best_primal_weight = iteration_limit_for_finding_best_primal_weight,
      iteration_limit = ITERATION_LIMIT,
      kkt_tolerance = kkt_tolerance,
      method,
    )
  end
end

main()
