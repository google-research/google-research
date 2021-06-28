# Implementation empirical bond length distributions, supporting pdf() functions.

using Combinatorics
using CSV
using DataFrames
using LinearAlgebra

using SmuUtilities

export BondLengthDistribution, AllBondLengthDistributions
export from_file!, from_arrays!, pdf
export add_from_files!

"""A single bond length distribution, holding a relationship between
  distance and value.
  It is assumed that the X values are uniformly distributed.
"""
mutable struct BondLengthDistribution
  bucket_size::Float32  # Distance between X values.
  min_distance::Float32
  max_distance::Float32
  pdf::Vector{Float32}  # Normalized prevalence values.
end

BondLengthDistribution() = BondLengthDistribution(0.0, 0.0, 0.0, [])

function from_file!(fname::AbstractString, distribution::BondLengthDistribution)::Bool
  df = CSV.File(fname) |> DataFrame
  return from_arrays!(df[!,1], df[!,2], distribution)
end

function from_arrays!(lengths::Vector, counts::Vector, distribution::BondLengthDistribution)::Bool
  min_diff, max_diff = extrema(diff(lengths))
  if abs(max_diff - min_diff) > 1.0e-05
    @warn("non uniform buckets $(min_diff) $(max_diff)")
    return false
  end
  distribution.bucket_size = (max_diff + min_diff) / 2.0  # For accuracy.
  distribution.min_distance = lengths[1]
  distribution.max_distance =  lengths[end] + distribution.bucket_size
  # During testing/development, fill in any missing values with 1
  distribution.pdf = Float32.(map((x)-> x==0 ? 1 : x, counts))
  normalize!(distribution.pdf, 1)
  return true
end

function pdf(distribution::BondLengthDistribution, distance::Real)::Float32
  distance <= distribution.min_distance && return 0.0
  distance >= distribution.max_distance && return 0.0

  idx = round(Int32, (distance - distribution.min_distance) / distribution.bucket_size) + 1
  idx > length(distribution.pdf) && return 0.0  # will happen due to fp inaccuracy
  return distribution.pdf[idx]
end

"""The AllBondLengthDistributions struct keeps a map from this type
  to the corresponding BondLengthDistribution.
"""
struct AtypesBtype
  type1::Int
  type2::Int
  btype::Int
end 

# functions to facilitate hashing
function Base.isequal(t1::AtypesBtype, t2::AtypesBtype)::Bool
  t1.type1 == t1.type1 && t1.type2 == t2.type2 && t1.btype == t2.btype
end

function Base.hash(atypes::AtypesBtype)::UInt64
  return 200 * atypes.type1 + 100 * atypes.type2 + atypes.btype
end

"""The main interface to bond length distributions.
  Holds a mapping from atom and bond types to a distribution, which si then
  used by pdf().
"""
mutable struct AllBondLengthDistributions
  # Whether or not the non-bonded types are read during construction 
  # from files.
  include_non_bonded::Bool
  # For each atom types/bond type combination, a distribution.
  distribution::Dict{AtypesBtype, BondLengthDistribution}
end
AllBondLengthDistributions() = AllBondLengthDistributions(true, Dict{AtypesBtype, BondLengthDistribution}())

"""Add a bond length distribution to `distributions` by reading data from `fname`.
Args:
Returns:
"""
function add_file!(fname::String, distributions::AllBondLengthDistributions, key::AtypesBtype)::Bool
  dist = BondLengthDistribution()
  if ! from_file!(fname, dist)
    @warn("cannot read distribution from $(fname)")
    return false
  end

  @debug("add_file got $key")
  distributions.distribution[key] = dist

  return true
end

"""Given a file name `stem` read all files that would be present for bond length
distributions and construct `distributions`. 

Note the very complex Iterators.product was inspired by
https://discourse.julialang.org/t/cleanest-way-to-generate-all-combinations-of-n-arrays/20127/8

No idea if this would be efficient for larger arrays, here it does not matter, and avoids a
doubly nested loop.

Args:
  stem: file name stem for the bond length distributions files. stem.6.1.6 would be for
    carbon-carbon single bonds
  distributions: The data read will be stored here.
Returns:
  true if successful
"""
function add_from_files!(stem::String, distributions::AllBondLengthDistributions)::Bool
  bstart  = distributions.include_non_bonded ? 0 : 1

  bond_types = [0, 1, 2, 3]

  atomic_numbers = [1, 6, 7, 8, 9]
  for (t1, t2) in filter(x->x[1] <= x[2], collect(Iterators.product(ntuple(i->atomic_numbers, 2)...))[:])
    for btype in bstart:3
      fname = "$stem.$(t1).$(btype).$(t2)"
      @info("Processing $fname")
      if ! isfile(fname)
        @warn("skipping non existent file $(fname)")
        continue
      end
      key = AtypesBtype(t1, t2, btype)
#     @debug("Adding to key $key")
      if ! add_file!(fname, distributions, key)
        @error("error adding $(fname)")
        return false
      end
    end
  end
  true
end

function pdf(distributions::AllBondLengthDistributions, type1::Integer, type2::Integer, btype::Integer, distance::Real)::Float32
  key = AtypesBtype(minmax(smu_atype_to_atomic_number(type1), smu_atype_to_atomic_number(type2))..., btype)
  value = get(distributions.distribution, key, nothing)
# @debug("value for $type1 $type2 $btype $value")
  value == nothing && return 0.0

  return pdf(value, distance)
end
