function find_best_primal_weight(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  primal_weight_trial_list::Vector{Float64},
  primal_weight_for_evaluation::Float64,
  step_size_for_evaluation::Float64,
)
  if params.method == ADMM
    # recycle the factorization between different primal weight values
    cached_factorization = build_ADMM_factorization(problem, params.verbosity)
  end
  best_kkt_error = Inf
  best_primal_weight = NaN
  for primal_weight in primal_weight_trial_list
    println("run for primal weight = $primal_weight")
    params.primal_weight = primal_weight
    if params.method == ADMM
      starting_point = zero_ADMM_iterate(
        length(problem.objective_vector),
        length(problem.right_hand_side),
      )
      solver_output =
        optimize_ADMM(params, problem, starting_point, cached_factorization)
    else
      solver_output = optimize(params, problem)
    end

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
