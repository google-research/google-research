using SparseArrays

abstract type RestartScheme end
struct NoRestarts <: RestartScheme
  # no attributes for NoRestarts class
end

struct FixedFrequencyRestarts <: RestartScheme
  restart_length::Int64
  always_reset_to_average::Bool # do we alway reset to average?
end

struct AdaptiveRestarts <: RestartScheme
  beta::Float64
  always_reset_to_average::Bool # do we alway reset to average?
end

mutable struct NormalizedDualityGaps
  average_distance_travelled::Float64
  average_gap::Float64
  current_distance_travelled::Float64
  current_gap::Float64
end

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

function RecoverPrimalDualMethodFromString(method_name::String)
  if method_name == "PDHG"
    return PDHG
  elseif method_name == "extragradient"
    return EXTRAGRADIENT
  elseif method_name == "ADMM"
    return ADMM
  else
    error("unsupported method choosen")
  end
end

"""
Logging while the algorithm is running.
"""
function log_iteration(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
  iteration_stats::DataFrames.DataFrame,
)

  Printf.@printf(
    "   %5d objectives=(%9g, %9g) norms=(%9g, %9g) res_norm=(%9g, %9g) kkt=%.2e\n",
    iteration_stats[:iteration][end],
    iteration_stats[:primal_objectives][end],
    iteration_stats[:dual_objectives][end],
    iteration_stats[:primal_solution_norms][end],
    iteration_stats[:dual_solution_norms][end],
    iteration_stats[:primal_delta_norms][end],
    iteration_stats[:dual_delta_norms][end],
    min(
      iteration_stats[:kkt_error_current_iterate][end],
      iteration_stats[:kkt_error_average_iterate][end],
    ),
  )
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

@enum SolutionStatus STATUS_OPTIMAL STATUS_ITERATION_LIMIT

"""
Output of the solver.
"""
struct PrimalDualOutput
  primal_solution::Vector{Float64}
  dual_solution::Vector{Float64}
  iteration_stats::DataFrames.DataFrame
  status::SolutionStatus
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

function compute_number_of_active_set_changes(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
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

function convert_problem_to_standard_form!(
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  m = problem.num_equalities
  n = size(problem.constraint_matrix, 1)
  if n > m
    slack_matrix = [spzeros(m, n - m); I]
    problem.constraint_matrix = [problem.constraint_matrix slack_matrix]
    problem.num_equalities = n
    problem.objective_vector = [problem.objective_vector; zeros(n - m)]
    problem.variable_lower_bound = [problem.variable_lower_bound; zeros(n - m)]
    problem.variable_upper_bound =
      [problem.variable_upper_bound; Inf * ones(n - m)]
    nvar = size(problem.objective_matrix, 1)
    problem.objective_matrix = [
      [problem.objective_matrix spzeros(nvar, n - m)]
      [spzeros(n - m, nvar) spzeros(n - m, n - m)]
    ]

    @show size(problem.constraint_matrix)
    @show size(problem.variable_lower_bound)
    @show size(problem.variable_upper_bound)
    @show size(problem.objective_vector)
    @show size(problem.objective_matrix)
  elseif m > n
    error("more equalities that rows in the constraint matrix")
  end
end
