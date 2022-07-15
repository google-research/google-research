import FirstOrderLp, Plots, CSV
using LaTeXStrings, DataFrames, Test
include("../src/PrimalDualMethods.jl")

simple_lp1 = FirstOrderLp.linear_programming_problem(
  [-Inf],  # variable_lower_bound
  [Inf],  # variable_upper_bound
  [0.0],  # objective_vector
  0.0,                 # objective_constant
  ones(1, 1),           # constraint_matrix
  [0.0],      # right_hand_side
  1,                     # num_equalities
)
simple_lp2 = FirstOrderLp.linear_programming_problem(
  [0.0, 0.0],  # variable_lower_bound
  [Inf, Inf],  # variable_upper_bound
  [2.0, 1.0],  # objective_vector
  0.0,                 # objective_constant
  ones(1, 2),           # constraint_matrix
  [1.0],      # right_hand_side
  1,                     # num_equalities
)

A = [
  -1.03324 -0.449087 0.322969 1.36308 1.46373 -1.63236
  -0.326885 0.471807 1.73327 -0.505794 0.543352 -1.2965
]
lp1_primal_optimal_solution = [0.0; 0.0; 0.0; 0.0; 1.0; 1.0]
b = A * lp1_primal_optimal_solution
lp1 = FirstOrderLp.linear_programming_problem(
  zeros(6),  # variable_lower_bound
  Inf * ones(6),  # variable_upper_bound
  [10.0, 10.0, 10.0, 10.0, 1.0, 1.0],  # objective_vector
  0.0,                 # objective_constant
  A,           # constraint_matrix
  b,      # right_hand_side
  2,      # num_equalities
)

A_B = A[:, end-1:end]
lp1_dual_optimal_solution = A_B' \ lp1.objective_vector[end-1:end]

restart_length = 25

for method_name in ["PDHG", "extragradient", "ADMM"]
  @testset "$method_name" begin
    for primal_weight in [1.0, 2.0]
      @testset "primal_weight=$primal_weight" begin
        params = PrimalDualOptimizerParameters(
          RecoverPrimalDualMethodFromString(method_name), # method
          0.2,  # step_size
          primal_weight,  # primal_weight
          1, # record every
          20,   # printing frequency
          false, # verbose
          2000,  # iteration limit
          NoRestarts(), # restart scheme
          nothing,
          1e-8,
        )

        ##############
        # simple lp1 #
        ##############
        # No restarts
        solver_output = optimize(params, simple_lp1, [1.0], [1.0])
        @test norm(solver_output.primal_solution) < 1e-7
        @test norm(solver_output.dual_solution) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL
        @test solver_output.iteration_stats.iteration[end] < 1500

        # Fixed frequency restarts
        params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
        solver_output = optimize(params, simple_lp1, [1.0], [1.0])
        @test norm(solver_output.primal_solution) < 1e-7
        @test norm(solver_output.dual_solution) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL
        @test solver_output.iteration_stats.iteration[end] < 600

        # Adaptive restarts
        params.restart_scheme = AdaptiveRestarts(exp(-1), true)
        solver_output = optimize(params, simple_lp1, [1.0], [1.0])
        @test norm(solver_output.primal_solution) < 1e-7
        @test norm(solver_output.dual_solution) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL
        @test solver_output.iteration_stats.iteration[end] < 600

        # Adaptive restarts
        params.restart_scheme = AdaptiveRestarts(exp(-1), false)
        solver_output = optimize(params, simple_lp1, [1.0], [1.0])
        @test norm(solver_output.primal_solution) < 1e-7
        @test norm(solver_output.dual_solution) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL
        @test solver_output.iteration_stats.iteration[end] < 600

        ##############
        # simple lp2 #
        ##############
        # No restarts
        solver_output = optimize(params, simple_lp2)
        @test norm(solver_output.primal_solution - [0.0, 1.0]) < 1e-3
        @test norm(solver_output.dual_solution - [1.0]) < 1e-3
        @test solver_output.status == STATUS_OPTIMAL

        # Fixed frequency restarts
        params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
        solver_output = optimize(params, simple_lp2)
        @test norm(solver_output.primal_solution - [0.0, 1.0]) < 1e-7
        @test norm(solver_output.dual_solution - [1.0]) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL

        # Adaptive restarts
        params.restart_scheme = AdaptiveRestarts(exp(-1), true)
        solver_output = optimize(params, simple_lp2)
        @test norm(solver_output.primal_solution - [0.0, 1.0]) < 1e-7
        @test norm(solver_output.dual_solution - [1.0]) < 1e-7
        @test solver_output.status == STATUS_OPTIMAL

        #######
        # lp1 #
        #######
        # No restarts
        params.kkt_tolerance = 1e-6
        params.restart_scheme = NoRestarts()
        solver_output = optimize(params, lp1)
        @test norm(
          solver_output.primal_solution - lp1_primal_optimal_solution,
        ) < 1e-2
        @test norm(solver_output.dual_solution - lp1_dual_optimal_solution) <
              1e-2
        # PDHG with NoRestarts hits the termination limit, so we have a
        # larger tolerance on the primal and dual solutions, and don't test
        # status.

        # Fixed frequency restarts
        params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
        solver_output = optimize(params, lp1)
        @test norm(
          solver_output.primal_solution - lp1_primal_optimal_solution,
        ) < 1e-5
        @test norm(solver_output.dual_solution - lp1_dual_optimal_solution) <
              1e-5
        @test solver_output.status == STATUS_OPTIMAL

        # Adaptive restarts
        params.restart_scheme = AdaptiveRestarts(exp(-1), true)
        solver_output = optimize(params, lp1)
        @test norm(
          solver_output.primal_solution - lp1_primal_optimal_solution,
        ) < 1e-5
        @test norm(solver_output.dual_solution - lp1_dual_optimal_solution) <
              1e-5
        @test solver_output.status == STATUS_OPTIMAL
      end
    end
  end
end
