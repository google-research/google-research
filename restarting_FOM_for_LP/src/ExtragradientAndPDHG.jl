include("ExtragradientAndPDHG_restart_scheme.jl")

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

function optimize_extragradient_or_pdhg(
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
          println("Found optimal solution after $iteration iterations")
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
        iteration >= iteration_limit ? STATUS_ITERATION_LIMIT : STATUS_OPTIMAL,
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
