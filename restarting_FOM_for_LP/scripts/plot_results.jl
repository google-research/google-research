using Latexify, DataFrames
import CSV, Plots

include("../src/shared_variables.jl")

function standard_plot_setup()
  Plots.plot(
    xlabel = "iterations",
    ylabel = "normalized duality gap",
    yaxis = :log,
    linestyle = :dash,
    legend = :topright,
  )
end

function get_residual(data::DataFrame)
  #return data.kkt_error_average_iterate
  return data.average_normalized_gap
  #return data.primal_delta_norms .+ data.dual_delta_norms
end

function select_data_before_iteration_limit(
  data::DataFrame,
  iteration_limit::Int64,
)
  indicies_before_iteration_limit = findall(data.iteration .<= iteration_limit)
  return data[indicies_before_iteration_limit, :]
end

function generic_restart_plot(;
  data::DataFrame,
  color::Symbol,
  markershape::Symbol,
  label::String,
  linestyle::Symbol,
  y_min::Float64,
  iteration_limit::Int64,
)
  data = select_data_before_iteration_limit(data, iteration_limit)
  residual = get_residual(data)
  # Plot the residuals
  Plots.plot!(
    data.iteration,
    residual,
    label = "",
    ylims = (y_min, maximum(residual)),
    color = color,
    linestyle = linestyle,
  )
  # Plot the restart points
  restart_indicies = findall(data.restart_occurred)
  Plots.plot!(
    data.iteration[restart_indicies],
    residual[restart_indicies],
    color = color,
    linealpha = 0.0,
    markershape = markershape,
    label = "",
    markerstrokewidth = 0,
    markerstrokecolor = :auto,
    markersize = 5.0,
  )
  # Plot the last active set change
  no_change_indicies = findall(data.number_of_active_set_changes .> 0)
  if length(no_change_indicies) > 0
    last_active_set_change_index = no_change_indicies[end]
    Plots.plot!(
      [data.iteration[last_active_set_change_index]],
      [residual[last_active_set_change_index]],
      color = color,
      markershape = :star5,
      linealpha = 0.0,
      label = "",
      markerstrokecolor = :auto,
      markersize = 7.0,
    )
  end
  # Artificial plot for the legend.
  restart_plt = Plots.plot!(
    data.iteration[restart_indicies][1:1],
    residual[restart_indicies][1:1],
    color = color,
    markershape = markershape,
    label = label,
    markerstrokecolor = :auto,
    linestyle = linestyle,
  )
  return restart_plt
end

function plot_no_restart_and_adaptive(
  results_directory::String,
  y_min::Float64,
  iteration_limit::Int64,
)
  no_restarts_df =
    CSV.read(joinpath(results_directory, "no_restarts.csv"), DataFrame)
  no_restarts_df =
    select_data_before_iteration_limit(no_restarts_df, iteration_limit)
  residual = no_restarts_df.current_normalized_gap
  Plots.plot!(
    no_restarts_df.iteration,
    residual,
    color = :black,
    label = "No restarts",
  )

  restarts_plt = generic_restart_plot(
    data = CSV.read(
      joinpath(results_directory, "adaptive_restarts.csv"),
      DataFrame,
    ),
    color = :blue,
    markershape = :circle,
    label = "Adaptive restarts",
    linestyle = :solid,
    y_min = y_min,
    iteration_limit = iteration_limit,
  )
  return restarts_plt
end

function plot_no_restart_adaptive_and_fixed_frequency_results(
  results_directory::String,
  restart_lengths::Vector{Int64},
  y_min::Float64,
  iteration_limit::Int64,
)
  # Figure out which restart lengths to plot
  df_restart_performance = DataFrame(
    restart_length = Int64[],
    first_approx_opt_index = Float64[],
    final_function_value = Float64[],
  )
  for i in 1:length(restart_lengths)
    restart_length = restart_lengths[i]
    fixed_frequency_df = CSV.read(
      joinpath(results_directory, "restart_length$(restart_length).csv"),
      DataFrame,
    )
    residuals = get_residual(fixed_frequency_df)
    approx_opt_indicies = findall(residuals .< y_min)
    if length(approx_opt_indicies) > 0
      first_approx_opt_index = approx_opt_indicies[1]
    else
      first_approx_opt_index = Inf
    end
    final_function_value = residuals[end]
    append!(
      df_restart_performance,
      DataFrame(
        restart_length = restart_length,
        first_approx_opt_index = first_approx_opt_index,
        final_function_value = final_function_value,
      ),
    )
  end
  sort!(
    df_restart_performance,
    [:first_approx_opt_index, :final_function_value],
    rev = (false, false),
  )
  # Pick three best restart lengths
  subset_of_restart_lengths = df_restart_performance.restart_length[1:3]
  sort!(subset_of_restart_lengths)

  standard_plot_setup()

  colors = [:red, :green, :purple, :orange, :pink]
  markers = [
    :dtriangle,
    :rect,
    :diamond,
    :hexagon,
    :cross,
    :xcross,
    :utriangle,
    :rtriangle,
    :ltriangle,
    :pentagon,
    :heptagon,
    :octagon,
    :vline,
    :hline,
  ]
  for i in 1:length(subset_of_restart_lengths)
    restart_length = subset_of_restart_lengths[i]
    fixed_frequency_df = CSV.read(
      joinpath(results_directory, "restart_length$(restart_length).csv"),
      DataFrame,
    )
    generic_restart_plot(;
      data = fixed_frequency_df,
      color = colors[i],
      markershape = markers[i],
      label = "Restart length = $restart_length",
      linestyle = :dot,
      y_min = y_min,
      iteration_limit = iteration_limit,
    )
  end

  restarts_plt =
    plot_no_restart_and_adaptive(results_directory, y_min, iteration_limit)
  return restarts_plt
end

function plot_dynamic_adaptive_and_no_restarts(
  results_directory::String,
  restart_lengths::Vector{Int64},
  y_min::Float64,
  iteration_limit::Int64,
)
  standard_plot_setup()
  plot_no_restart_and_adaptive(results_directory, y_min, iteration_limit)

  dynamic_adaptive_restarts_df = CSV.read(
    joinpath(results_directory, "dynamic_adaptive_restarts.csv"),
    DataFrame,
  )
  residual = get_residual(dynamic_adaptive_restarts_df)
  flexible_restarts_plt = Plots.plot!(
    dynamic_adaptive_restarts_df.iteration,
    residual,
    label = "Flexible restarts",
    ylims = (y_min, maximum(residual)),
    color = :blue,
    linestyle = :dot,
  )
  return flexible_restarts_plt
end

function first_iteration_to_hit_tolerance(
  df::DataFrame,
  target_tolerance::Float64,
)
  indicies_below_tolerance = findall(
    min.(df.kkt_error_average_iterate, df.kkt_error_current_iterate) .<=
    target_tolerance,
  )
  if length(indicies_below_tolerance) > 0
    return df.iteration[indicies_below_tolerance[1]]
  else
    return Inf
  end
end


function create_dictionary_of_iterations_to_hit_tolerance(
  problem_name::String,
  results_directory::String,
  restart_lengths::Vector{Int64},
  target_tolerance::Float64,
)

  dictionary_hits = Dict()
  dictionary_hits["problem_name"] = problem_name
  dictionary_hits["no_restarts"] = first_iteration_to_hit_tolerance(
    CSV.read(joinpath(results_directory, "no_restarts.csv"), DataFrame),
    target_tolerance,
  )
  dictionary_hits["adaptive_restarts"] = first_iteration_to_hit_tolerance(
    CSV.read(joinpath(results_directory, "adaptive_restarts.csv"), DataFrame),
    target_tolerance,
  )
  dictionary_hits["flexible_restarts"] = first_iteration_to_hit_tolerance(
    CSV.read(
      joinpath(results_directory, "dynamic_adaptive_restarts.csv"),
      DataFrame,
    ),
    target_tolerance,
  )
  best_fixed_frequency_iterations = Inf
  for restart_length in restart_lengths
    fixed_frequency_df = CSV.read(
      joinpath(results_directory, "restart_length$(restart_length).csv"),
      DataFrame,
    )
    fixed_frequency_iterations =
      first_iteration_to_hit_tolerance(fixed_frequency_df, target_tolerance)
    if fixed_frequency_iterations < best_fixed_frequency_iterations
      best_fixed_frequency_iterations = fixed_frequency_iterations
    end
  end
  dictionary_hits["best_fixed_frequency"] = best_fixed_frequency_iterations

  return dictionary_hits
end

results_directory = ARGS[1]
@assert length(ARGS) == 1


function main()
  y_min = 1e-7
  plot_iteration_limit_dict = Dict(
    "qap10" => ITERATION_LIMIT,
    "qap15" => ITERATION_LIMIT,
    # manually make nug08-3rd shorter so the plot is interesting
    "nug08-3rd" => 10000,
    "nug20" => ITERATION_LIMIT,
  )

  for problem_name in ALL_PROBLEM_NAMES
    subdirectory = joinpath(results_directory, problem_name)
    restarts_plt = plot_no_restart_adaptive_and_fixed_frequency_results(
      subdirectory,
      RESTART_LENGTHS_DICT[problem_name],
      y_min,
      ITERATION_LIMIT,
    )
    Plots.savefig(
      restarts_plt,
      joinpath(subdirectory, "$(problem_name)-adaptive-residuals.pdf"),
    )
    flexible_plt = plot_dynamic_adaptive_and_no_restarts(
      subdirectory,
      RESTART_LENGTHS_DICT[problem_name],
      y_min,
      ITERATION_LIMIT,
    )
    Plots.savefig(
      flexible_plt,
      joinpath(subdirectory, "$(problem_name)-flexible-residuals.pdf"),
    )
  end

  ####################
  # table of results #
  ####################
  target_tolerance = 1e-6
  df_hit_tolerance = DataFrames.DataFrame(
    problem_name = String[],
    no_restarts = Float64[],
    best_fixed_frequency = Float64[],
    adaptive_restarts = Float64[],
    flexible_restarts = Float64[],
  )
  for problem_name in ALL_PROBLEM_NAMES
    subdirectory = joinpath(results_directory, problem_name)
    push!(
      df_hit_tolerance,
      create_dictionary_of_iterations_to_hit_tolerance(
        problem_name,
        subdirectory,
        RESTART_LENGTHS_DICT[problem_name],
        target_tolerance,
      ),
    )
  end

  latex_table_file = open(joinpath(results_directory, "latex_table.txt"), "w")
  write(latex_table_file, latexify(df_hit_tolerance, env = :table))
  close(latex_table_file)
end

main()
