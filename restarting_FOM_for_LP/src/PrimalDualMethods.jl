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
include("methods_enum.jl")
include("shared_functions.jl")
include("ExtragradientAndPDHG.jl")
include("ADMM.jl")

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
  if params.method == ADMM
    return optimize_ADMM(
      params,
      problem,
      AdmmIterate(
        initial_primal_solution,
        initial_primal_solution,
        zeros(length(initial_primal_solution)),
        initial_dual_solution,
      ),
    )
  else
    return optimize_extragradient_or_pdhg(
      params,
      problem,
      initial_primal_solution,
      initial_dual_solution,
    )
  end
end

function optimize(
  params::PrimalDualOptimizerParameters,
  problem::FirstOrderLp.QuadraticProgrammingProblem,
)
  if problem.num_equalities != size(problem.constraint_matrix, 1)
    error("Implementation only supports equality constraints")
  end
  if sum(abs.(problem.objective_matrix)) > 0.0
    error("Implementation only supports LP")
  end
  current_primal_solution = zeros(length(problem.variable_lower_bound))
  current_dual_solution = zeros(length(problem.right_hand_side))

  return optimize(
    params,
    problem,
    current_primal_solution,
    current_dual_solution,
  )
end
