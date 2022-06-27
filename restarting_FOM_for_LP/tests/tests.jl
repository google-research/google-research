import FirstOrderLp, Plots, CSV
using LaTeXStrings, DataFrames, Test
include("../src/PrimalDualMethods.jl")

simple_lp = FirstOrderLp.linear_programming_problem(
  [-Inf],  # variable_lower_bound
  [Inf],  # variable_upper_bound
  [0.0],  # objective_vector
  0.0,                 # objective_constant
  ones(1, 1),           # constraint_matrix
  [0.0],      # right_hand_side
  1,                     # num_equalities
)
restart_length = 25

for method_name in ["PDHG", "extragradient"]
  @testset "$method_name" begin
    params = PrimalDualOptimizerParameters(
      RecoverPrimalDualMethodFromString(method_name), # method
      0.2,  # step_size
      1.0,  # primal_weight
      1, # record every
      20,   # printing frequency
      false, # verbose
      600,  # iteration limit
      NoRestarts(), # restart scheme
      TwoDimensionalSubspace([1.0, 0.0], [0.0, 1.0]),
      1e-10,
    )
    # No restarts
    solver_output = optimize(params, simple_lp, [1.0], [1.0])
    @test norm(solver_output.primal_solution) < 1e-3
    @test norm(solver_output.dual_solution) < 1e-3

    # Fixed frequency restarts
    params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
    solver_output = optimize(params, simple_lp, [1.0], [1.0])
    @test norm(solver_output.primal_solution) < 1e-8
    @test norm(solver_output.dual_solution) < 1e-8

    # Adaptive restarts
    params.restart_scheme = AdaptiveRestarts(exp(-1), true)
    solver_output = optimize(params, simple_lp, [1.0], [1.0])
    @test norm(solver_output.primal_solution) < 1e-8
    @test norm(solver_output.dual_solution) < 1e-8
  end
end
