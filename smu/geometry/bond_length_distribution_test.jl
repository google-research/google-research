# Test framework for BondLenghtDistribution

module TestBondLengthDistribution

using Test

export test_single_distribution

include("bond_length_distribution.jl")

# using BondLengthDistributions

function triangular_distribution(min_dist::Real,
                                 dist_max_value::Real,
                                 max_dist::Real)  # ::Tuple{Vector{Float32}, Vector{Float32}}
  x_resolution = 1000


  population = zeros(Float32, x_resolution)
  x_extent = max_dist - min_dist
  peak_index = round(Int32, (dist_max_value - min_dist) / x_extent * x_resolution) + 1
  dy = 1.0 / peak_index
  for i in 1:peak_index
    population[i] = i * dy
  end

  dy = 1.0 / (x_resolution - peak_index)
  for i in peak_index:x_resolution
    population[i] = (1.0 - (i - peak_index) * dy)
  end

  dx = x_extent / x_resolution
  distances = collect(min_dist:dx:max_dist)

  return (distances, population)
end

function test_single_distribution()::Bool
  min_dist = 1.0
  mode_point = 1.4
  max_dist = 2.0
  (lengths, counts) = triangular_distribution(min_dist, mode_point, max_dist)
  distribution = BondLengthDistributions.BondLengthDistribution()

  @test BondLengthDistributions.from_arrays!(lengths, counts, distribution)
  @test BondLengthDistributions.pdf(distribution, 0.99 * min_dist) == 0.0
  @test BondLengthDistributions.pdf(distribution, 1.001 * max_dist) == 0.0

  @test BondLengthDistributions.pdf(distribution, 1.001 * min_dist) > 0.0
  @test BondLengthDistributions.pdf(distribution, 0.99  * max_dist) > 0.0

  maxval = BondLengthDistributions.pdf(distribution, mode_point)
  @test BondLengthDistributions.pdf(distribution, 0.98 * mode_point) < maxval
# @test BondLengthDistributions.pdf(distribution, 1.02 * mode_point) < maxval
  true
end


end  # module TestBondLengthDistribution
