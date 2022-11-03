import FirstOrderLp
import Base.isnan
using LinearAlgebra, Printf


"""
Solves the subproblem

min c' x + (1/2) * || x ||^2
s.t. A x = b

returning the primal and dual solution using a factorization of

[ I  A^T ]
[ A  0 ]
"""
function prox_onto_linear_system(
  c::Vector{Float64},
  b::Vector{Float64},
  cached_factorization::Factorization,
)
  sol = cached_factorization \ [c; b]
  if isnan_or_inf(sol)
    @show norm(c)
    @show norm(b)
    error("nan solution obtained when projecting onto linear system")
  end
  return sol[1:length(c)], -sol[length(c)+1:end]
end

mutable struct AdmmIterate
  x_U::Vector{Float64}
  x_V::Vector{Float64}
  y::Vector{Float64} # dual for the ADMM constraint
  # dual of the constraint matrix of the linear program
  constraint_matrix_duals::Vector{Float64}
end

function zero_ADMM_iterate(num_vars::Int64, num_cons::Int64)
  return AdmmIterate(
    zeros(num_vars),
    zeros(num_vars),
    zeros(num_vars),
    zeros(num_cons),
  )
end

function isnan_or_inf(vec::Vector{Float64})
  t = norm(vec)
  return isnan(t) || isinf(t)
end

function isnan_or_inf(z::AdmmIterate)
  return isnan_or_inf(z.x_U) ||
         isnan_or_inf(z.x_V) ||
         isnan_or_inf(z.y) ||
         isnan_or_inf(z.constraint_matrix_duals)
end

function print_z_details(z::AdmmIterate)
  @printf "||x_U||=%.2e ||x_V||=%.2e ||y||=%.2e constraint_duals=%.2e\n" norm(z.x_U) norm(z.x_V) norm(z.y) norm(z.constraint_matrix_duals)
end

mutable struct ADMM_average
  it::AdmmIterate
  counter::Int64
end

function update_average(z_avg::ADMM_average, z::AdmmIterate)
  z_avg.counter += 1
  alpha = 1 / z_avg.counter
  it = z_avg.it
  it.x_U = (1 - alpha) * it.x_U + alpha * z.x_U
  it.x_V = (1 - alpha) * it.x_V + alpha * z.x_V
  it.y = (1 - alpha) * it.y + alpha * z.y
  it.constraint_matrix_duals =
    (1 - alpha) * it.constraint_matrix_duals + alpha * z.constraint_matrix_duals
end

function reset_average(z_avg::ADMM_average)
  z_avg.counter = 0
end

function compute_stats(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration::Int64,
  two_dimensional_subspace::Union{TwoDimensionalSubspace,Nothing},
  z::AdmmIterate,
  z_avg::ADMM_average,
  primal_delta_norm::Float64,
  dual_delta_norm::Float64,
  restart_occurred::Bool,
  number_of_active_set_changes::Int64,
  normalized_gaps::NormalizedDualityGaps,
)
  average_iterate_convergence_info = compute_convergence_information(
    problem,
    z_avg.it.x_V,
    z_avg.it.constraint_matrix_duals,
  )
  current_iterate_convergence_info =
    compute_convergence_information(problem, z.x_V, z.constraint_matrix_duals)

  if nothing != two_dimensional_subspace
    z_vec = [z.x_V; z.x_U; z.y]
    current_subspace_coordinate1 =
      dot(z_vec, two_dimensional_subspace.basis_vector1)
    current_subspace_coordinate2 =
      dot(z_vec, two_dimensional_subspace.basis_vector2)
    z_vec_avg = [z_avg.x_V; z_avg.x_U; z_avg.y]
    average_subspace_coordinate1 =
      dot(z_vec_avg, two_dimensional_subspace.basis_vector1)
    average_subspace_coordinate2 =
      dot(z_vec_avg, two_dimensional_subspace.basis_vector2)
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
    primal_solution_norms = norm([z.x_U; z.x_V]),
    dual_solution_norms = norm(z.y),
    primal_delta_norms = primal_delta_norm,
    dual_delta_norms = dual_delta_norm,
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

function compute_admm_normalized_duality_gap(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  z::AdmmIterate,
  z_prev::AdmmIterate,
  eta::Float64,
)
  center_point = [z.x_V; z.y]
  trust_region_objective_vector =
    [-(z.y + problem.objective_vector); z.x_V - z.x_U]
  norm_weights = [
    eta * ones(length(problem.objective_vector))
    1 / eta * ones(length(z.x_U))
  ]
  distance_travelled = norm([z.x_V; z.y] - [z_prev.x_V; z_prev.y])
  trust_region_solution = FirstOrderLp.solve_bound_constrained_trust_region(
    center_point,
    -trust_region_objective_vector,
    [problem.variable_lower_bound; -Inf * ones(length(z.y))],
    [problem.variable_upper_bound; Inf * ones(length(z.y))],
    norm_weights,
    distance_travelled,
    false,
  )
  return distance_travelled, -trust_region_solution.value / distance_travelled
end

function compute_admm_normalized_duality_gap(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  z_avg::ADMM_average,
  z::AdmmIterate,
  z_prev::AdmmIterate,
  eta::Float64,
)
  distance_travelled_by_avg, avg_normalized_duality_gap =
    compute_admm_normalized_duality_gap(problem, z_avg.it, z_prev, eta)
  distance_travelled_by_current, current_normalized_duality_gap =
    compute_admm_normalized_duality_gap(problem, z, z_prev, eta)

  return NormalizedDualityGaps(
    distance_travelled_by_avg,
    avg_normalized_duality_gap,
    distance_travelled_by_current,
    current_normalized_duality_gap,
  )
end

function build_ADMM_factorization(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  verbose::Bool;
  LINEAR_SYSTEM_PERTURBATION::Float64 = 1e-8, # to ensure linear system is always factorizable
)
  if verbose
    print("factorizing constraint matrix ... ")
  end
  A = problem.constraint_matrix
  A_T = copy(problem.constraint_matrix')
  cached_factorization = ldlt([[I A_T]; [A -LINEAR_SYSTEM_PERTURBATION * I]])
  if verbose
    println("factorization complete")
  end
  return cached_factorization
end

function optimize_ADMM(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  starting_point::AdmmIterate,
)
  cached_factorization = build_ADMM_factorization(problem, params.verbosity)
  return optimize_ADMM(params, problem, starting_point, cached_factorization)
end

function optimize_ADMM(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  starting_point::AdmmIterate,
  cached_factorization::Factorization,
)

  z = starting_point
  z_last_restart = deepcopy(z)
  z_avg = ADMM_average(deepcopy(z), 0)

  stats = create_stats_data_frame()
  restart_scheme = params.restart_scheme
  eta = params.primal_weight
  if params.verbosity
    println("eta = $eta")
  end
  constraint_matrix_transpose = copy(problem.constraint_matrix')
  number_of_active_set_changes = 0
  last_restart_normalized_duality_gap = Inf
  iteration = 0
  primal_delta_norm = dual_delta_norm = 0.0
  while true
    restart_occurred = false
    # store stats and log
    terminate = false
    if iteration >= params.iteration_limit
      if params.verbosity
        println("Iteration limit reached")
      end
      terminate = true
    end

    gaps = compute_admm_normalized_duality_gap(
      problem,
      z_avg,
      z,
      z_last_restart,
      eta,
    )

    best_normalized_duality_gap = gaps.average_gap
    reset_to_average = true
    if isa(restart_scheme, FixedFrequencyRestarts) ||
       isa(restart_scheme, AdaptiveRestarts)
      if gaps.average_gap > gaps.current_gap &&
         !restart_scheme.always_reset_to_average
        best_normalized_duality_gap = gaps.current_gap
        reset_to_average = false
      end
    end

    if isa(restart_scheme, FixedFrequencyRestarts)
      do_restart = z_avg.counter >= restart_scheme.restart_length
    elseif isa(restart_scheme, AdaptiveRestarts) && iteration >= 1
      do_restart =
        (iteration == 1) ||
        best_normalized_duality_gap <=
        restart_scheme.beta * last_restart_normalized_duality_gap
      if do_restart
        last_restart_normalized_duality_gap = best_normalized_duality_gap
      end
    else
      do_restart = false
    end


    if do_restart
      if params.verbosity && isa(restart_scheme, AdaptiveRestarts)
        print("restart at iteration $iteration")
        if isa(restart_scheme, AdaptiveRestarts)
          print(" with normalized_gap = $best_normalized_duality_gap")
        end
        if !reset_to_average
          print(" to the current iterate")
        end
        println("")
      end
      if reset_to_average
        z = deepcopy(z_avg.it)
      end
      z_last_restart = deepcopy(z)
      reset_average(z_avg)
      restart_occurred = true
    end

    store_stats =
      mod(iteration, params.record_every) == 0 || terminate || do_restart
    print_stats =
      params.verbosity && (
        mod(iteration, params.record_every * params.print_every) == 0 ||
        terminate
      )

    if store_stats
      this_iteration_stats = compute_stats(
        problem,
        iteration,
        params.two_dimensional_subspace,
        z,
        z_avg,
        primal_delta_norm,
        dual_delta_norm,
        restart_occurred,
        number_of_active_set_changes,
        gaps,
      )
      kkt_error = min(
        this_iteration_stats.kkt_error_current_iterate[1],
        this_iteration_stats.kkt_error_average_iterate[1],
      )
      if kkt_error <= params.kkt_tolerance
        if params.verbosity
          println("Found optimal solution at iteration $iteration")
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
        z.x_V,
        z.constraint_matrix_duals,
        stats,
        iteration >= params.iteration_limit ? STATUS_ITERATION_LIMIT :
        STATUS_OPTIMAL,
      )
    end
    iteration += 1

    z_new = deepcopy(z)
    # min_{x_U} c^T z.x_V + 0.5 * eta * || x_U - z.x_V - z.y/\eta  ||^2 s.t. A x = b
    dual_target = z.x_V + z.y / eta
    dual_residual =
      dual_target - z.x_U +
      constraint_matrix_transpose * z.constraint_matrix_duals / eta
    primal_residual =
      problem.right_hand_side - problem.constraint_matrix * z.x_U
    delta_x_U, delta_scaled_dual = prox_onto_linear_system(
      dual_residual,
      primal_residual,
      cached_factorization,
    )
    delta_constraint_matrix_duals = delta_scaled_dual * eta
    z_new.x_U += delta_x_U
    z_new.constraint_matrix_duals += delta_constraint_matrix_duals

    # min_{x_V} 0.5 * eta * || x_V - z_new.x_U + z.y/\eta  ||^2 s.t. 0 <= x_V
    z_new.x_V = z_new.x_U - z.y / eta - (1 / eta) * problem.objective_vector
    FirstOrderLp.project_primal!(z_new.x_V, problem)
    z_new.y = z.y - eta * (z_new.x_U - z_new.x_V)
    z_hat = deepcopy(z_new)
    z_hat.y = z.y - eta * (z_new.x_U - z.x_V)
    update_average(z_avg, z_hat)

    primal_delta_norm = norm([z_new.x_U - z.x_U; z_new.x_V - z.x_V])
    dual_delta_norm = norm(z_new.y - z.y)

    zero_dual = zeros(length(z_new.y))
    number_of_active_set_changes = compute_number_of_active_set_changes(
      problem,
      z_new.x_U,
      zero_dual,
      z_new.x_U - z.x_U,
      zero_dual,
    )
    if isnan_or_inf(z_new)
      print(iteration)
      print("z:")
      print_z_details(z)
      print("z_new:")
      print_z_details(z_new)
      error("nan or inf in z_new")
    end

    z = z_new
  end
end
