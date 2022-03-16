abstract type RestartScheme end
struct NoRestarts <: RestartScheme

end

struct FixedFrequencyRestarts <: RestartScheme
  restart_length::Int64
  always_reset_to_average::Bool # do we alway reset to average?
end

struct AdaptiveRestarts <: RestartScheme
  beta::Float64
  always_reset_to_average::Bool # do we alway reset to average?
end

mutable struct RestartInfo
  last_restart_primal_solution::Vector{Float64}
  last_restart_dual_solution::Vector{Float64}
  normalized_duality_gap::Float64
end

function initialize_restart_info(primal_size::Int64, dual_size::Int64)
  last_restart_primal_solution = zeros(primal_size)
  last_restart_dual_solution = zeros(dual_size)
  normalized_duality_gap = Inf
  return RestartInfo(
    last_restart_primal_solution,
    last_restart_dual_solution,
    normalized_duality_gap,
  )
end

function compute_normalized_duality_gap(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
  primal_weight::Float64,
  distance_travelled::Float64,
)
  local_duality_gap = FirstOrderLp.bound_optimal_objective(
    problem,
    primal_solution,
    dual_solution,
    ones(length(primal_solution)) * primal_weight, # primal_norm_params
    ones(length(dual_solution)) / primal_weight, # dual_norm_params
    distance_travelled,
    FirstOrderLp.EUCLIDEAN_NORM, # norm
    solve_approximately = false,
  )
  return FirstOrderLp.get_gap(local_duality_gap) / distance_travelled
end

mutable struct NormalizedDualityGaps
  average_distance_travelled::Float64
  average_gap::Float64
  current_distance_travelled::Float64
  current_gap::Float64
end

function compute_normalized_duality_gaps(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  restart_info::RestartInfo,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  primal_weight::Float64,
)
  distance_travelled_by_avg = norm([
    avg_primal_solution - restart_info.last_restart_primal_solution
    avg_dual_solution - restart_info.last_restart_dual_solution
  ])
  avg_normalized_duality_gap = compute_normalized_duality_gap(
    problem,
    avg_primal_solution,
    avg_dual_solution,
    primal_weight,
    distance_travelled_by_avg,
  )
  distance_travelled_by_current = norm([
    current_primal_solution - restart_info.last_restart_primal_solution
    current_dual_solution - restart_info.last_restart_dual_solution
  ])
  current_normalized_duality_gap = compute_normalized_duality_gap(
    problem,
    current_primal_solution,
    current_dual_solution,
    primal_weight,
    distance_travelled_by_current,
  )
  return NormalizedDualityGaps(
    distance_travelled_by_avg,
    avg_normalized_duality_gap,
    distance_travelled_by_current,
    current_normalized_duality_gap,
  )
end

function select_best_iterate(
  gaps::NormalizedDualityGaps,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  always_reset_to_average::Bool,
)
  restart_to_current = false
  if !always_reset_to_average
    if gaps.current_gap < gaps.average_gap
      restart_to_current = true
    end
  end
  if restart_to_current
    candidate_normalized_duality_gap = gaps.current_gap
    candidate_primal_solution = current_primal_solution
    candidate_dual_solution = current_dual_solution
  else
    candidate_normalized_duality_gap = gaps.average_gap
    candidate_primal_solution = avg_primal_solution
    candidate_dual_solution = avg_dual_solution
  end

  return candidate_normalized_duality_gap,
  candidate_primal_solution,
  candidate_dual_solution
end

function run_restart_scheme(
  restart_scheme::RestartScheme,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  primal_weight::Float64,
  solution_weighted_avg::FirstOrderLp.SolutionWeightedAverage,
  restart_info::RestartInfo,
  verbose::Bool,
)

  do_fixed_frequency_restart =
    isa(restart_scheme, FixedFrequencyRestarts) &&
    solution_weighted_avg.sum_primal_solutions_count >=
    restart_scheme.restart_length
  consider_adaptive_restart = isa(restart_scheme, AdaptiveRestarts)

  primal_avg, dual_avg = FirstOrderLp.compute_average(solution_weighted_avg)
  gaps = compute_normalized_duality_gaps(
    problem,
    restart_info,
    current_primal_solution,
    current_dual_solution,
    primal_avg,
    dual_avg,
    primal_weight,
  )

  do_restart = false
  if do_fixed_frequency_restart || consider_adaptive_restart
    candidate_normalized_duality_gap,
    candidate_primal_solution,
    candidate_dual_solution = select_best_iterate(
      gaps,
      current_primal_solution,
      current_dual_solution,
      primal_avg,
      dual_avg,
      restart_scheme.always_reset_to_average,
    )
    if do_fixed_frequency_restart
      do_restart = true
    elseif consider_adaptive_restart
      if candidate_normalized_duality_gap <=
         restart_scheme.beta * restart_info.normalized_duality_gap
        do_restart = true
      end
    end
  end
  if do_restart
    if verbose
      println("restart algorithm after $(solution_weighted_avg.sum_primal_solutions_count) iterations")
    end
    restart_info.normalized_duality_gap = candidate_normalized_duality_gap
    current_primal_solution .= candidate_primal_solution
    current_dual_solution .= candidate_dual_solution
    restart_info.last_restart_primal_solution .= current_primal_solution
    restart_info.last_restart_dual_solution .= current_dual_solution
    FirstOrderLp.reset_solution_weighted_average(solution_weighted_avg)
  end

  return do_restart, gaps
end
