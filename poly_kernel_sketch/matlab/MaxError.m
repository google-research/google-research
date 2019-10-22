% Matlab script to plot max error among set of vectors as function of sketch size and input dimension.

fixed_dimension = 100
fixed_sketch_dim = 100
kernel_degree = 2  % makes little difference likely
num_repeat = 100
num_plot_points = 20

sketch_types = ["Tensor Sketch", "Tensorized Random Projection"]

plot_by_dimension(fixed_sketch_dim, kernel_degree, sketch_types, num_repeat, num_plot_points);
plot_by_sketch_size(fixed_dimension, kernel_degree, sketch_types, num_repeat, num_plot_points);

function plot_by_dimension(fixed_sketch_dim, kernel_degree, sketch_types, num_repeat, num_plot_points)
for sketch_type = sketch_types
  disp(sketch_type)
  step = round(4*sqrt(fixed_sketch_dim) / num_plot_points);
  dims = zeros(num_plot_points, 1);
  mean_err = zeros(num_plot_points, 1);
  std_err = zeros(num_plot_points, 1);
  for i = 1:num_plot_points
    dims(i) = step * i;
    [mean_err(i), std_err(i)] = sketch_error(dims(i), kernel_degree, fixed_sketch_dim, sketch_type, num_repeat);
  end
  errorbar(dims, mean_err, std_err);
  hold on;
end
xlabel("Input dimension");
ylabel("Max error");
legend(sketch_types);
saveas(gcf, '../../figures/max_err_vs_dim.png');
hold off;
end

function plot_by_sketch_size(fixed_dimension, kernel_degree, sketch_types, num_repeat, num_plot_points)
for sketch_type = sketch_types
  disp(sketch_type)
  step = fixed_dimension * fixed_dimension / num_plot_points;
  sketch_dims = zeros(num_plot_points, 1);
  mean_err = zeros(num_plot_points, 1);
  std_err = zeros(num_plot_points, 1);
  for i = 1:num_plot_points
    sketch_dims(i) = step * i;
    [mean_err(i), std_err(i)] = sketch_error(fixed_dimension, kernel_degree, sketch_dims(i), sketch_type, num_repeat);
  end
  errorbar(sketch_dims, mean_err, std_err);
  hold on;
end
xlabel("Sketch dimension");
ylabel("Max error");
legend(sketch_types);
saveas(gcf, '../../figures/max_err_vs_sketch_size.png');
hold off;
end

function [mean_err, std_err] = sketch_error(dimension, kernel_degree, sketch_dim, sketch_type, num_repeat)
errors = zeros(num_repeat, 1);
for i = 1:num_repeat
  errors(i) = max_sketch_error(dimension, kernel_degree, sketch_dim, sketch_type);
end
mean_err = mean(errors);
std_err = std(errors);
end

function out = max_sketch_error(dimension, kernel_degree, sketch_dim, sketch_type)
% standard basis vectors
data = eye(dimension);
if sketch_type == "Tensor Sketch"
  data_sketch = FFT_CountSketch_k_Naive(data, kernel_degree, sketch_dim);
elseif sketch_type == "Tensorized Random Projection"
  data_sketch = OUR_SKETCH(data, kernel_degree, sketch_dim);
else
  error("Unknown sketch type %s", sketch_type);
end
sketched_kernel = data_sketch * transpose(data_sketch);
kernel_error = sketched_kernel - eye(dimension);
out = max(abs(kernel_error(:)));
end


