# This is a simplified version of the primal-dual hybrid gradient
# and extragradient implementation, of the package FirstOrderLp
# (https://github.com/google-research/FirstOrderLp.jl)
# aimed to be used for experiments with restarts.

import Printf
import LinearAlgebra
import Random
import DataFrames
const norm = LinearAlgebra.norm
const dot = LinearAlgebra.dot
const randn = Random.randn
include("restart_scheme.jl")

"""
Information for recording iterates on two dimensional subspace.
"""
struct TwoDimensionalSubspace
  """
  Subspace basis vector 1
  """
  basis_vector1::Vector{Float64}
  """
  Subspace basis vector 2, should be orthognal to vector 1
  """
  basis_vector2::Vector{Float64}
end


@enum PrimalDualMethod PDHG EXTRAGRADIENT

function RecoverPrimalDualMethodFromString(method_name::String)
  if method_name == "PDHG"
    return PDHG
  elseif method_name == "extragradient"
    return EXTRAGRADIENT
  else
    error("unsupported method choosen")
  end
end


"""
A PrimalDualOptimizerParameters struct specifies the parameters for solving the saddle
point formulation of an problem using primal-dual hybrid gradient.

It solves a problem of the form (see quadratic_programmming.jl in FirstOrderLp)
minimize objective_vector' * x

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end, :]

     variable_lower_bound <= x <= variable_upper_bound

We use notation from Chambolle and Pock, "On the ergodic convergence rates of a
first-order primal-dual algorithm"
(http://www.optimization-online.org/DB_FILE/2014/09/4532.pdf).
That paper doesn't explicitly use the terminology "primal-dual hybrid gradient"
but their Theorem 1 is analyzing PDHG. In this file "Theorem 1" without further
reference refers to that paper.

Our problem is equivalent to the saddle point problem:
    min_x max_y L(x, y)
where
    L(x, y) = y' K x + g(x) - h*(y)
    K = -constraint_matrix
    g(x) = objective_vector' x if variable_lower_bound <= x <=
                                                            variable_upper_bound
                               otherwise infinity
    h*(y) = -right_hand_side' y if y[(num_equalities + 1):end] >= 0
                                otherwise infinity

Note that the places where g(x) and h*(y) are infinite effectively limits the
domain of the min and max. Therefore there's no infinity in the code.

We parametrize the primal and dual step sizes (tau and sigma in Chambolle and
Pock) as:
    primal_step_size = step_size / primal_weight
    dual_step_size = step_size * primal_weight.
The algoritm converges if
    primal_stepsize * dual_stepsize * norm(contraint_matrix)^2 < 1.
"""
mutable struct PrimalDualOptimizerParameters
  """
  method = PDHG or extragradient
  """
  method::PrimalDualMethod

  """
  Constant step size used in the algorithm. If nothing is specified, the solver
  computes a provably correct step size.
  """
  step_size::Union{Float64,Nothing}
  """
  Weight relating primal and dual step sizes.
  """
  primal_weight::Float64
  """
  Records iteration stats to csv every record_every iterations.
  During these iteration restarts may also be performed.
  """
  record_every::Int64
  """
  If verbose is true then prints iteration stats every print_every time that
  iteration information is recorded.
  """
  print_every::Int64
  """
  If true a line of debugging info is printed every printing_every
  iterations.
  """
  verbosity::Bool
  """
  Number of loop iterations to run. Must be postive.
  """
  iteration_limit::Int64
  """
  Which restart scheme should we use
  """
  restart_scheme::RestartScheme
  """
  A subspace where projections of iterates are recorded.
  If the value is nothing then the values will be missing from the iteration
  stats.
  """
  two_dimensional_subspace::Union{TwoDimensionalSubspace,Nothing}

  kkt_tolerance::Float64
end

function create_stats_data_frame()
  return DataFrames.DataFrame(
    # Record the current iteration.
    iteration = Int64[],
    # Primal objectives of the iterates; the ith entry corresponds to the primal
    # objective of the ith iterate.
    primal_objectives = Float64[],
    # Dual objectives of the iterates; the ith entry corresponds to the dual
    # objective of the ith iterate.
    dual_objectives = Float64[],
    # Primal norms of the iterates; the ith entry corresponds to the primal
    # norm of the ith iterate.
    primal_solution_norms = Float64[],
    # Dual norms of the iterates; the ith entry corresponds to the dual
    # norm of the ith iterate.
    dual_solution_norms = Float64[],
    # Primal delta norms of the iterates; the ith entry corresponds to the
    # primal delta norm of the ith iterate.
    primal_delta_norms = Float64[],
    # Dual delta norms of the iterates; the ith entry corresponds to the dual
    # delta norm of the ith iterate.
    dual_delta_norms = Float64[],
    # First coordinate of a subspace that the current iterates are projected
    # onto.
    current_subspace_coordinate1 = Float64[],
    # Second coordinate of a subspace that the current iterates are projected
    # onto.
    current_subspace_coordinate2 = Float64[],
    # First coordinate of a subspace that the average iterates are projected
    # onto.
    average_subspace_coordinate1 = Float64[],
    # Second coordinate of a subspace that the average iterates are projected
    # onto.
    average_subspace_coordinate2 = Float64[],
    # Did a restart occur at this iteration
    restart_occurred = Bool[],
    # Number of variables that moved in or out of the active set at this
    # iteration
    number_of_active_set_changes = Int64[],
    kkt_error_current_iterate = Float64[],
    kkt_error_average_iterate = Float64[],
    average_normalized_gap = Float64[],
    current_normalized_gap = Float64[],
  )
end

"""
Output of the solver.
"""
struct PrimalDualOutput
  primal_solution::Vector{Float64}
  dual_solution::Vector{Float64}
  iteration_stats::DataFrames.DataFrame
end

function compute_convergence_information(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
)
  dual_stats =
    FirstOrderLp.compute_dual_stats(problem, primal_solution, dual_solution)
  primal_objective = FirstOrderLp.primal_obj(problem, primal_solution)
  dual_objective = dual_stats.dual_objective
  primal_residual =
    FirstOrderLp.compute_primal_residual(problem, primal_solution)
  kkt_error = sqrt(
    sum(primal_residual .^ 2) +
    sum(dual_stats.dual_residual .^ 2) +
    (primal_objective - dual_objective)^2,
  )
  return (
    primal_objective = primal_objective,
    dual_objective = dual_objective,
    kkt_error = kkt_error,
  )
end

"""
Computes statistics for the current iteration. The arguments primal_delta and
dual_delta correspond to the difference between the last two primal and dual
iterates, respectively.
"""
function compute_stats(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration::Int64,
  two_dimensional_subspace::Union{TwoDimensionalSubspace,Nothing},
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
  restart_occurred::Bool,
  number_of_active_set_changes::Int64,
  normalized_gaps::NormalizedDualityGaps,
)
  if restart_occurred
    primal_avg, dual_avg = primal_solution, dual_solution
  else
    primal_avg, dual_avg = FirstOrderLp.compute_average(solution_weighted_avg)
  end
  current_iterate_convergence_info =
    compute_convergence_information(problem, primal_solution, dual_solution)
  average_iterate_convergence_info =
    compute_convergence_information(problem, primal_avg, dual_avg)

  kkt_error_average_iterate = Inf

  if nothing != two_dimensional_subspace
    z = [primal_solution; dual_solution]
    current_subspace_coordinate1 =
      dot(z, two_dimensional_subspace.basis_vector1)
    current_subspace_coordinate2 =
      dot(z, two_dimensional_subspace.basis_vector2)
    z_avg = [primal_avg; dual_avg]
    average_subspace_coordinate1 =
      dot(z_avg, two_dimensional_subspace.basis_vector1)
    average_subspace_coordinate2 =
      dot(z_avg, two_dimensional_subspace.basis_vector2)
  else
    current_subspace_coordinate1 = NaN
    current_subspace_coordinate2 = NaN
    average_subspace_coordinate1 = NaN
    average_subspace_coordinate2 = NaN
  end

  return DataFrames.DataFrame(
    iteration = iteration,
    primal_objectives = current_iterate_convergence_info.primal_objective,
    dual_objectives = current_iterate_convergence_info.dual_objective,
    primal_solution_norms = norm(primal_solution),
    dual_solution_norms = norm(dual_solution),
    primal_delta_norms = norm(primal_delta),
    dual_delta_norms = norm(dual_delta),
    current_subspace_coordinate1 = current_subspace_coordinate1,
    current_subspace_coordinate2 = current_subspace_coordinate2,
    average_subspace_coordinate1 = average_subspace_coordinate1,
    average_subspace_coordinate2 = average_subspace_coordinate2,
    restart_occurred = restart_occurred,
    number_of_active_set_changes = number_of_active_set_changes,
    kkt_error_current_iterate = current_iterate_convergence_info.kkt_error,
    kkt_error_average_iterate = average_iterate_convergence_info.kkt_error,
    average_normalized_gap = normalized_gaps.average_gap,
    current_normalized_gap = normalized_gaps.current_gap,
  )
end

"""
Logging while the algorithm is running.
"""
function log_iteration(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration_stats::DataFrames.DataFrame,
)

  Printf.@printf(
    "   %5d objectives=(%9g, %9g) norms=(%9g, %9g) res_norm=(%9g, %9g)\n",
    iteration_stats[:iteration][end],
    iteration_stats[:primal_objectives][end],
    iteration_stats[:dual_objectives][end],
    iteration_stats[:primal_solution_norms][end],
    iteration_stats[:dual_solution_norms][end],
    iteration_stats[:primal_delta_norms][end],
    iteration_stats[:dual_delta_norms][end],
  )
end

function take_pdhg_step(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  primal_weight::Float64,
  step_size::Float64,
)
  # The next lines compute the primal portion of the PDHG algorithm:
  # argmin_x [g(x) + current_dual_solution' K x
  #          + (0.5 * primal_weight / step_size)
  #             * norm(x - current_primal_solution)^2]
  # See Sections 2-3 of Chambolle and Pock and the comment above
  # PrimalDualOptimizerParameters.
  # This minimization is easy to do in closed form since it can be separated
  # into independent problems for each of the primal variables. The
  # projection onto the primal feasibility set comes from the closed form
  # for the above minimization and the cases where g(x) is infinite - there
  # isn't officially any projection step in the algorithm.
  primal_gradient = FirstOrderLp.compute_primal_gradient(
    problem,
    current_primal_solution,
    current_dual_solution,
  )
  next_primal =
    current_primal_solution .- primal_gradient * (step_size / primal_weight)
  FirstOrderLp.project_primal!(next_primal, problem)

  # The next two lines compute the dual portion:
  # argmin_y [H*(y) - y' K (2.0*next_primal - current_primal_solution)
  #           + (0.5 / (primal_weight * step_size))
  #.             * norm(y-current_dual_solution)^2]
  dual_gradient = FirstOrderLp.compute_dual_gradient(
    problem,
    2.0 * next_primal - current_primal_solution,
  )
  next_dual =
    current_dual_solution .+ dual_gradient * (step_size * primal_weight)
  FirstOrderLp.project_dual!(next_dual, problem)

  return next_primal, next_dual
end

function proximal_step(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  current_primal_gradient::Vector{Float64},
  current_dual_gradient::Vector{Float64},
  primal_weight::Float64,
  step_size::Float64,
)
  prox_primal =
    current_primal_solution -
    (step_size / primal_weight) * current_primal_gradient
  FirstOrderLp.project_primal!(prox_primal, problem)
  prox_dual =
    current_dual_solution + (step_size * primal_weight) * current_dual_gradient
  FirstOrderLp.project_dual!(prox_dual, problem)
  return prox_primal, prox_dual
end

function take_extragradient_step(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  primal_weight::Float64,
  step_size::Float64,
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
)
  current_primal_gradient = FirstOrderLp.compute_primal_gradient(
    problem,
    current_primal_solution,
    current_dual_solution,
  )
  current_dual_gradient =
    FirstOrderLp.compute_dual_gradient(problem, current_primal_solution)
  test_primal_solution, test_dual_solution = proximal_step(
    problem,
    current_primal_solution,
    current_dual_solution,
    current_primal_gradient,
    current_dual_gradient,
    primal_weight,
    step_size,
  )
  test_primal_gradient = FirstOrderLp.compute_primal_gradient(
    problem,
    test_primal_solution,
    test_dual_solution,
  )
  test_dual_gradient =
    FirstOrderLp.compute_dual_gradient(problem, test_primal_solution)
  FirstOrderLp.add_to_solution_weighted_average(
    solution_weighted_avg,
    test_primal_solution,
    test_dual_solution,
    1.0,
  )
  return proximal_step(
    problem,
    current_primal_solution,
    current_dual_solution,
    test_primal_gradient,
    test_dual_gradient,
    primal_weight,
    step_size,
  )
end

function compute_number_of_active_set_changes(
  problem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  primal_delta::Vector{Float64},
  dual_delta::Vector{Float64},
)
  number_of_active_set_changes = 0
  for idx in 1:length(current_primal_solution)
    if primal_delta[idx] != 0.0 && (
      current_primal_solution[idx] == problem.variable_lower_bound[idx] ||
      current_primal_solution[idx] == problem.variable_upper_bound[idx]
    )
      number_of_active_set_changes += 1
    end
  end
  for idx in (problem.num_equalities+1):length(current_dual_solution)
    if dual_delta[idx] != 0.0 && current_dual_solution[idx] == 0.0
      number_of_active_set_changes += 1
    end
  end
  return number_of_active_set_changes
end


"""
`optimize(params::PrimalDualOptimizerParameters,
          problem::QuadraticProgrammingProblem)`

Solves a linear program using primal-dual hybrid gradient or
extragradient. If the step_size
specified in params is negative, picks a step size that ensures
step_size^2 * norm(constraint_matrix)^2 < 1,
a condition that guarantees provable convergence.

# Arguments
- `params::PrimalDualOptimizerParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.

# Returns
A PrimalDualOutput struct containing the solution found.
"""
function optimize(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  initial_primal_solution::Vector{Float64},
  initial_dual_solution::Vector{Float64},
)
  primal_size = length(problem.variable_lower_bound)
  dual_size = length(problem.right_hand_side)

  primal_weight = params.primal_weight

  if isnothing(params.step_size)
    desired_relative_error = 0.2
    maximum_singular_value, number_of_power_iterations =
      FirstOrderLp.estimate_maximum_singular_value(
        problem.constraint_matrix,
        probability_of_failure = 0.001,
        desired_relative_error = desired_relative_error,
      )
    step_size = (1 - desired_relative_error) / maximum_singular_value
  else
    step_size = params.step_size
  end
  if params.verbosity
    Printf.@printf(
      "Step size: %9g, Primal weight: %9g, method: %s\n",
      step_size,
      primal_weight,
      params.method
    )
  end

  restart_scheme = params.restart_scheme
  restart_info = initialize_restart_info(primal_size, dual_size)
  solution_weighted_avg =
    FirstOrderLp.initialize_solution_weighted_average(primal_size, dual_size)
  # Difference between current_primal_solution and last primal iterate, i.e.
  # x_k - x_{k-1}.
  primal_delta = zeros(primal_size)
  # Difference between current_dual_solution and last dual iterate, i.e. y_k -
  # y_{k-1}
  dual_delta = zeros(dual_size)
  iteration_limit = params.iteration_limit
  stats = create_stats_data_frame()

  current_primal_solution = initial_primal_solution
  current_dual_solution = initial_dual_solution

  restart_occurred = false
  number_of_active_set_changes = 0
  iteration = 0
  while true
    # store stats and log
    terminate = false
    if iteration >= iteration_limit
      if params.verbosity
        println("Iteration limit reached")
      end
      terminate = true
    end
    do_fixed_frequency_restart =
      isa(restart_scheme, FixedFrequencyRestarts) &&
      solution_weighted_avg.sum_primal_solutions_count >=
      restart_scheme.restart_length
    store_stats =
      mod(iteration, params.record_every) == 0 ||
      terminate ||
      do_fixed_frequency_restart
    print_stats =
      params.verbosity && (
        mod(iteration, params.record_every * params.print_every) == 0 ||
        terminate
      )
    if store_stats
      if iteration > 0
        restart_occurred, normalized_gap_info = run_restart_scheme(
          restart_scheme,
          problem,
          current_primal_solution,
          current_dual_solution,
          primal_weight,
          solution_weighted_avg,
          restart_info,
          params.verbosity,
        )
      else
        restart_occurred = false
        normalized_gap_info = NormalizedDualityGaps(NaN, NaN, NaN, NaN)
      end
      this_iteration_stats = compute_stats(
        problem,
        iteration,
        params.two_dimensional_subspace,
        current_primal_solution,
        current_dual_solution,
        solution_weighted_avg,
        primal_delta,
        dual_delta,
        restart_occurred,
        number_of_active_set_changes,
        normalized_gap_info,
      )
      kkt_error = min(
        this_iteration_stats.kkt_error_current_iterate[1],
        this_iteration_stats.kkt_error_average_iterate[1],
      )
      if kkt_error <= params.kkt_tolerance
        if params.verbosity
          println("Found optimal solution")
        end
        terminate = true
      end
    end
    if store_stats
      append!(stats, this_iteration_stats)
    end
    if print_stats
      log_iteration(problem, this_iteration_stats)
    end

    if terminate
      return PrimalDualOutput(
        current_primal_solution,
        current_dual_solution,
        stats,
      )
    end
    iteration += 1

    if params.method == PDHG
      next_primal, next_dual = take_pdhg_step(
        problem,
        current_primal_solution,
        current_dual_solution,
        primal_weight,
        step_size,
      )
      FirstOrderLp.add_to_solution_weighted_average(
        solution_weighted_avg,
        next_primal,
        next_dual,
        1.0,
      )
    elseif params.method == EXTRAGRADIENT
      next_primal, next_dual = take_extragradient_step(
        problem,
        current_primal_solution,
        current_dual_solution,
        primal_weight,
        step_size,
        solution_weighted_avg,
      )
    else
      error("unknown method")
    end
    # Update deltas and iterates
    primal_delta = next_primal .- current_primal_solution
    dual_delta = next_dual .- current_dual_solution

    number_of_active_set_changes = compute_number_of_active_set_changes(
      problem,
      current_primal_solution,
      current_dual_solution,
      primal_delta,
      dual_delta,
    )

    # update iterates
    current_primal_solution = next_primal
    current_dual_solution = next_dual
  end
end

function optimize(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  current_primal_solution = zeros(length(problem.variable_lower_bound))
  current_dual_solution = zeros(length(problem.right_hand_side))

  return optimize(
    params,
    problem,
    current_primal_solution,
    current_dual_solution,
  )
end
