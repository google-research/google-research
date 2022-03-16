import FirstOrderLp, Plots, CSV
using LaTeXStrings, DataFrames
include("../src/PrimalDualMethods.jl")

@assert length(ARGS) == 2

output_dir = ARGS[1]
method = RecoverPrimalDualMethodFromString(ARGS[2])

mkpath(output_dir)
println("Creating 1D bilinear problem\n")
""" Returns a 1D bilinear problem.
"""
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

# Current iterate with no restarts
params = PrimalDualOptimizerParameters(
  method, # method
  0.2,  # step_size
  1.0,  # primal_weight
  1, # record every
  20,   # printing frequency
  true, # verbose
  51,  # iteration limit
  NoRestarts(), # restart scheme
  TwoDimensionalSubspace([1.0, 0.0], [0.0, 1.0]),
  1e-10,
)

# No restarts
print("About to start optimizing\n")
solver_output = optimize(params, simple_lp, [1.0], [1.0])

no_restarts_csv_path = joinpath(output_dir, "no_restarts.csv")
CSV.write(no_restarts_csv_path, solver_output.iteration_stats)

# Fixed frequency restarts
params.restart_scheme = FixedFrequencyRestarts(restart_length, true)
print("About to start optimizing\n")
solver_output = optimize(params, simple_lp, [1.0], [1.0])

fixed_frequency_csv_path = joinpath(output_dir, "fixed_frequency_results.csv")
CSV.write(fixed_frequency_csv_path, solver_output.iteration_stats)


# Plots
fixed_frequency_df = CSV.read(fixed_frequency_csv_path, DataFrame)
no_restarts_df = CSV.read(no_restarts_csv_path, DataFrame)

Plots.plot(
  [0.0],
  [0.0],
  label = "Solution",
  markershape = :star,
  linealpha = 0.0,
  markerstrokewidth = 0.0,
  markersize = 6,
  color = "black",
  legend = :bottomleft,
)

iterates_plt = Plots.plot!(
  no_restarts_df.current_subspace_coordinate1,
  no_restarts_df.current_subspace_coordinate2,
  label = "No restarts (current iterate)",
  markershape = :diamond,
  markersize = 8,
  markerstrokewidth = 0.0,
  linealpha = 0.0,
  markerstrokealpha = 0.0,
  color = "red",
  xlabel = L"x_1",
  ylabel = L"y_1",
  markerstrokecolor = "red",
)

Plots.plot!(
  no_restarts_df.average_subspace_coordinate1[2:end],
  no_restarts_df.average_subspace_coordinate2[2:end],
  label = "No restarts (average iterate)",
  markershape = :cross,
  linealpha = 0.0,
  markerstrokewidth = 2.0,
  color = "red",
)

Plots.plot!(
  fixed_frequency_df.average_subspace_coordinate1[2:end],
  fixed_frequency_df.average_subspace_coordinate2[2:end],
  label = "Fixed frequency restarts (average iterate)",
  markershape = :xcross,
  linealpha = 0.0,
  markerstrokealpha = 0.0,
  markerstrokewidth = 2.0,
  color = "blue",
)

Plots.plot!(
  fixed_frequency_df.current_subspace_coordinate1,
  fixed_frequency_df.current_subspace_coordinate2,
  label = "Fixed frequency restarts (current iterate)",
  markershape = :circle,
  markersize = 4,
  linealpha = 0.0,
  markerstrokewidth = 0.0,
  markerstrokecolor = "blue",
  color = "blue",
)

Plots.savefig(iterates_plt, joinpath(output_dir, "bilinear-iterates.pdf"))
